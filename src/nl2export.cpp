// NL2ModelExport.dll - capture the geometry NoLimits 2 renders, for export to OBJ/glTF.
//
// Track geometry in NL2 lives only in GPU buffers (pooled D3D11 vertex/index buffers, no CPU copy).
// This DLL, injected into nolimits2stm.exe, does what the engine's own VertexBuffer::lock() does
// (staging copy + Map) but for everything drawn during a capture window:
//
//   1. On load it creates a throw-away D3D11 device + swap chain to learn the vtable addresses of
//      IDXGISwapChain::Present/Present1 and ID3D11DeviceContext::Draw*, then patches those vtable slots
//      (shared by every device/context of the process, so the game's calls land in our hooks).
//   2. "CAPTURE <frames> [full] <dir>" over TCP 127.0.0.1:15153 arms a capture. For the next <frames> frames every
//      draw on the game's immediate context is recorded (with "full": frustum culling, software occlusion and
//      LOD distance are switched off first, so one frame holds the whole park at its finest detail) with its full input-assembler state
//      (vertex buffers/strides/offsets, index buffer/format/offset, topology, counts, instancing,
//      input layout, VS/PS shader, first PS texture, VS constant buffers).
//   3. At the end of the last captured frame (inside Present = game's render thread) every referenced
//      buffer is copied to a STAGING buffer, mapped and written to <dir>/buffers/<id>.bin.
//   4. The engine's own nl3d VertexBuffer/IndexBuffer objects are found by scanning the heap for their
//      vtables; their vertex-element layout is written to <dir>/nl3d_buffers.json so the converter knows
//      what each byte means (position/normal/uv...).
//
// Output: <dir>/draws.json, <dir>/buffers/*.bin, <dir>/nl3d_buffers.json. Use tools/nl2_capture_to_obj.py.
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <atomic>
#include <algorithm>
#include <psapi.h>

// ------------------------------------------------------------------ logging
static FILE* g_log;
static CRITICAL_SECTION g_logCs;
static void Log(const char* fmt, ...) {
  if (!g_log) return;
  EnterCriticalSection(&g_logCs);
  SYSTEMTIME st; GetLocalTime(&st);
  fprintf(g_log, "[%02d:%02d:%02d.%03d] ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
  va_list ap; va_start(ap, fmt); vfprintf(g_log, fmt, ap); va_end(ap);
  fputc('\n', g_log); fflush(g_log);
  LeaveCriticalSection(&g_logCs);
}

// ------------------------------------------------------------------ game constants (nolimits2stm.exe build 0x696F4197)
static const uintptr_t VT_D3D11_VERTEXBUFFER = 0x140cdaaa8;   // Direct3D11VertexBuffer@win32@nl3d
static const uintptr_t VT_D3D11_INDEXBUFFER  = 0x140cdaa80;   // Direct3D11IndexBuffer@win32@nl3d
static const uintptr_t G_D3D_CONTEXT = 0x14102dc50;   // ID3D11DeviceContext* (= Direct3D11RendererContext singleton @0x14102dc30, +0x20)
static const uintptr_t G_D3D_DEVICE  = 0x14102dc90;   // ID3D11Device*
static const uintptr_t FN_ENGINE_PRESENT = 0x14087f1b0; // Direct3D11Renderer::present (calls IDXGISwapChain::Present)
static const uint8_t ENGINE_PRESENT_PROLOGUE[15] = {0x40,0x57,0x48,0x83,0xEC,0x60,0x48,0xC7,0x44,0x24,0x20,0xFE,0xFF,0xFF,0xFF};
// "full" captures (whole park in one frame): culling + LOD overrides.
//   NLMain is the engine's ApplicationCallbackHandler; vtable slot 0x88 is its per-frame cull callback
//   (main.cpp, called by Engine::render with the eye ViewState before the cull worker is kicked).
//   ViewState +0x14/+0x18 = LOD distance offset/scale (every LOD pick is dist*scale+offset < threshold[i], -1 = hidden),
//   +0x3ec = bit mask of active frustum planes (0 = every frustum test passes).
//   0x140fd4201 = occlusion mode (1 = software occlusion with occluders, 2 = none).
//   0x1407a3d90 / 0x1407a4340 = software-occlusion box tests (return char visible), patched to "mov al,1; ret".
static const uintptr_t VT_NLMAIN = 0x140bef1d0;
static const int NLMAIN_CULL_SLOT = 0x88 / 8;
static const uintptr_t G_OCCLUSION_MODE = 0x140fd4201;
static const uintptr_t FN_SOB_TESTS[2] = {0x1407a3d90, 0x1407a4340};
static const uint8_t SOB_PROLOGUE[3] = {0x40, 0x55, 0x56};

// ------------------------------------------------------------------ hook plumbing
typedef HRESULT (STDMETHODCALLTYPE* PFN_Present)(IDXGISwapChain*, UINT, UINT);
typedef HRESULT (STDMETHODCALLTYPE* PFN_Present1)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
typedef void (STDMETHODCALLTYPE* PFN_DrawIndexed)(ID3D11DeviceContext*, UINT, UINT, INT);
typedef void (STDMETHODCALLTYPE* PFN_Draw)(ID3D11DeviceContext*, UINT, UINT);
typedef void (STDMETHODCALLTYPE* PFN_DrawIndexedInstanced)(ID3D11DeviceContext*, UINT, UINT, UINT, INT, UINT);
typedef void (STDMETHODCALLTYPE* PFN_DrawInstanced)(ID3D11DeviceContext*, UINT, UINT, UINT, UINT);

typedef HRESULT (STDMETHODCALLTYPE* PFN_Map)(ID3D11DeviceContext*, ID3D11Resource*, UINT, D3D11_MAP, UINT, D3D11_MAPPED_SUBRESOURCE*);
typedef void (STDMETHODCALLTYPE* PFN_Unmap)(ID3D11DeviceContext*, ID3D11Resource*, UINT);
typedef void (STDMETHODCALLTYPE* PFN_UpdateSubresource)(ID3D11DeviceContext*, ID3D11Resource*, UINT, const D3D11_BOX*, const void*, UINT, UINT);
static PFN_Map o_Map; static PFN_Unmap o_Unmap; static PFN_UpdateSubresource o_UpdateSubresource;
// The immediate context exposes several interfaces (ID3D11DeviceContext, ...1, ...2, ...3, ...4), each with its own
// inline vtable. We patch every distinct one and keep the originals per vtable.
struct CtxOrig { void** vt; void* fn[64]; };
static CtxOrig g_ctxOrig[8]; static int g_ctxOrigN = 0;
// Wrapper context: the engine's context global is pointed at this object while capturing.
// Every vtable slot is a tiny stub that swaps in the real context and jumps to the real method,
// except the handful we intercept. Nothing inside d3d11/dxgi is modified.
struct WrapCtx { void** vtbl; ID3D11DeviceContext* real; };
static WrapCtx g_wrap;
static const int WRAP_SLOTS = 160;
static inline ID3D11DeviceContext* RealOf(void* c) { return ((WrapCtx*)c)->real; }
static inline void* OrigFor(void* c, int slot) { return (*(void***)RealOf(c))[slot]; }
static PFN_Present o_Present; static PFN_Present1 o_Present1;
static PFN_DrawIndexed o_DrawIndexed; static PFN_Draw o_Draw;
static PFN_DrawIndexedInstanced o_DrawIndexedInstanced; static PFN_DrawInstanced o_DrawInstanced;

static void* PatchVtable(void** vtbl, int index, void* hook) {
  void* orig = vtbl[index];
  if (orig == hook) return orig;   // already ours (should not happen)
  DWORD old; VirtualProtect(&vtbl[index], sizeof(void*), PAGE_EXECUTE_READWRITE, &old);
  vtbl[index] = hook;
  VirtualProtect(&vtbl[index], sizeof(void*), old, &old);
  return orig;
}

// ------------------------------------------------------------------ re-entrancy diagnostics
static thread_local int t_depth = 0;
static std::atomic<int> g_btLogged{0};
static void LogBacktrace(const char* where) {
  if (g_btLogged.fetch_add(1) >= 3) return;
  void* frames[48]; USHORT n = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
  Log("RE-ENTRY in %s (depth %d) - call chain:", where, t_depth);
  for (USHORT i = 0; i < n; ++i) {
    HMODULE m = nullptr; char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)frames[i], &m) && m) {
      GetModuleFileNameA(m, name, MAX_PATH);
      const char* b = strrchr(name, '\\');
      Log("   #%02u %p %s+0x%llx", i, frames[i], b ? b + 1 : name, (unsigned long long)((uint8_t*)frames[i] - (uint8_t*)m));
    } else Log("   #%02u %p", i, frames[i]);
  }
}
struct DepthGuard {
  bool deep;
  DepthGuard(const char* where) { deep = ++t_depth > 8; if (deep) LogBacktrace(where); }
  ~DepthGuard() { --t_depth; }
};
static bool g_optNoCb = false, g_optNoScan = false, g_optInstOnly = false;

// ------------------------------------------------------------------ capture state
struct DrawRec {
  int kind;                 // 0 DrawIndexed, 1 Draw, 2 DrawIndexedInstanced, 3 DrawInstanced
  UINT count, instances, startIndex, startVertex, startInstance; INT baseVertex;
  ID3D11Buffer* vb[4]; UINT stride[4], offset[4];
  ID3D11Buffer* ib; DXGI_FORMAT ibFmt; UINT ibOffset;
  D3D11_PRIMITIVE_TOPOLOGY topo;
  void* layout; void* vs; void* ps; void* tex0; ID3D11Buffer* vscb[4];
  void* tex[4];             // pixel-shader textures (resources) in slots 0..3
  int psBlob[4];            // pixel-shader constant buffers 0..3 (shadowed contents), -1 = none
  int texSet;               // bits 0-7: PS texture slots (re)bound since the previous draw; bits 8-15: since the last PSSetShader
                            // (slots never bound for a shader are stale leftovers the shader does not read)
  int cbBlob[4];            // index into g_cbBlobs (shadowed constant-buffer contents at draw time), -1 = unknown
  int instBlob;             // index into g_cbBlobs: bytes of vertex slot 1 used by this draw (instance/per-object data), -1 = none
  int frame;
};
// Constant-buffer shadowing: the game writes per-draw data (world matrices...) into constant buffers with
// Map/Unmap or UpdateSubresource. We keep a CPU copy of the latest contents per buffer while capturing.
struct PendingMap { void* ptr; UINT size; };   // size = bytes to copy at Unmap
static std::map<ID3D11Resource*, PendingMap> g_pendingMaps;
static std::map<ID3D11Buffer*, std::string> g_cbShadow;
static std::vector<std::string> g_cbBlobs;
static std::map<std::string, int> g_cbBlobIndex;
// Constant buffers, plus dynamic vertex buffers (the per-draw instance stream in slot 1 is rewritten before every draw).
static bool IsConstantBuffer(ID3D11Resource* r, UINT* size) {
  D3D11_RESOURCE_DIMENSION dim; r->GetType(&dim);
  if (dim != D3D11_RESOURCE_DIMENSION_BUFFER) return false;
  D3D11_BUFFER_DESC d; ((ID3D11Buffer*)r)->GetDesc(&d);
  if (size) *size = d.ByteWidth;
  if (d.BindFlags & D3D11_BIND_CONSTANT_BUFFER) return true;
  return (d.BindFlags & D3D11_BIND_VERTEX_BUFFER) && d.Usage == D3D11_USAGE_DYNAMIC && d.ByteWidth <= (4u << 20);
}
static CRITICAL_SECTION g_cs;
static std::atomic<int> g_framesLeft{0};
static int g_frameIndex = 0;
static std::string g_outDir;
static std::vector<DrawRec> g_draws;
static std::set<std::string> g_seen;                 // dedupe identical draws across frames
static std::set<ID3D11Buffer*> g_buffers;            // referenced buffers (we hold a ref each)
static std::set<ID3D11Resource*> g_textures;         // referenced textures (we hold a ref each)
static HANDLE g_doneEvent;
static std::string g_result;

static unsigned g_texFresh = 0, g_texSinceShader = 0;   // render thread only
static void RecordDraw(ID3D11DeviceContext* ctx, int kind, UINT count, UINT inst, UINT startIdx, INT baseV, UINT startV, UINT startInst) {
  if (g_framesLeft.load() <= 0) return;
  if (ctx->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE) return;
  if (g_optInstOnly && kind < 2) { g_texFresh = 0; return; }   // "inst": only instanced draws (bolts/nuts), for fast park-wide harvesting
  DrawRec d{}; d.texSet = (int)((g_texFresh & 0xff) | ((g_texSinceShader & 0xff) << 8)); g_texFresh = 0;
  d.kind = kind; d.count = count; d.instances = inst; d.startIndex = startIdx; d.baseVertex = baseV;
  d.startVertex = startV; d.startInstance = startInst; d.frame = g_frameIndex;
  ctx->IAGetVertexBuffers(0, 4, d.vb, d.stride, d.offset);
  ctx->IAGetIndexBuffer(&d.ib, &d.ibFmt, &d.ibOffset);
  ctx->IAGetPrimitiveTopology(&d.topo);
  ID3D11InputLayout* il = nullptr; ctx->IAGetInputLayout(&il); d.layout = il; if (il) il->Release();
  ID3D11VertexShader* vs = nullptr; ctx->VSGetShader(&vs, nullptr, nullptr); d.vs = vs; if (vs) vs->Release();
  ID3D11PixelShader* ps = nullptr; ctx->PSGetShader(&ps, nullptr, nullptr); d.ps = ps; if (ps) ps->Release();
  ID3D11ShaderResourceView* srvs[4] = {};
  ctx->PSGetShaderResources(0, 4, srvs);
  ID3D11Resource* texr[4] = {};
  for (int i = 0; i < 4; ++i) {
    if (!srvs[i]) continue;
    srvs[i]->GetResource(&texr[i]); d.tex[i] = texr[i]; srvs[i]->Release();
  }
  d.tex0 = d.tex[0];
  ctx->VSGetConstantBuffers(0, 4, d.vscb);
  ID3D11Buffer* pscb[4] = {};
  ctx->PSGetConstantBuffers(0, 4, pscb);
  std::string cbKey;
  EnterCriticalSection(&g_cs);
  for (int i = 0; i < 4; ++i) {
    d.cbBlob[i] = -1;
    if (!d.vscb[i]) continue;
    auto it = g_cbShadow.find(d.vscb[i]);
    if (it == g_cbShadow.end()) continue;
    auto bi = g_cbBlobIndex.find(it->second);
    if (bi == g_cbBlobIndex.end()) { bi = g_cbBlobIndex.emplace(it->second, (int)g_cbBlobs.size()).first; g_cbBlobs.push_back(it->second); }
    d.cbBlob[i] = bi->second;
  }
  for (int i = 0; i < 4; ++i) {
    d.psBlob[i] = -1;
    if (!pscb[i]) continue;
    auto it = g_cbShadow.find(pscb[i]);
    if (it == g_cbShadow.end()) continue;
    auto bi = g_cbBlobIndex.find(it->second);
    if (bi == g_cbBlobIndex.end()) { bi = g_cbBlobIndex.emplace(it->second, (int)g_cbBlobs.size()).first; g_cbBlobs.push_back(it->second); }
    d.psBlob[i] = bi->second;
  }
  d.instBlob = -1;
  if (d.vb[1] && d.stride[1]) {
    auto it = g_cbShadow.find(d.vb[1]);
    if (it != g_cbShadow.end()) {
      size_t a = d.offset[1] + (size_t)d.startInstance * d.stride[1], n = (size_t)std::max(1u, d.instances) * d.stride[1];
      if (a + n <= it->second.size()) {
        std::string blob = it->second.substr(a, n);
        auto bi = g_cbBlobIndex.find(blob);
        if (bi == g_cbBlobIndex.end()) { bi = g_cbBlobIndex.emplace(blob, (int)g_cbBlobs.size()).first; g_cbBlobs.push_back(blob); }
        d.instBlob = bi->second;
      }
    }
  }
  LeaveCriticalSection(&g_cs);
  for (int i = 0; i < 4; ++i) { char t[16]; snprintf(t, sizeof t, "|%d", d.cbBlob[i]); cbKey += t; }
  { char t[16]; snprintf(t, sizeof t, "|i%d", d.instBlob); cbKey += t; }
  for (int i = 0; i < 4; ++i) { char t[40]; snprintf(t, sizeof t, "|p%d:%p", d.psBlob[i], d.tex[i]); cbKey += t; }
  char key[512];
  snprintf(key, sizeof key, "%d|%u|%u|%u|%d|%u|%u|%p:%u:%u|%p:%u:%u|%p:%d:%u|%d",
           kind, count, inst, startIdx, baseV, startV, startInst, (void*)d.vb[0], d.stride[0], d.offset[0],
           (void*)d.vb[1], d.stride[1], d.offset[1], (void*)d.ib, (int)d.ibFmt, d.ibOffset, (int)d.topo);
  EnterCriticalSection(&g_cs);
  bool fresh = g_seen.insert(std::string(key) + cbKey).second;
  if (fresh) {
    for (int i = 0; i < 4; ++i) if (d.vb[i]) { if (g_buffers.insert(d.vb[i]).second) d.vb[i]->AddRef(); }
    if (d.ib && g_buffers.insert(d.ib).second) d.ib->AddRef();
    for (int i = 0; i < 4; ++i) if (texr[i] && g_textures.insert(texr[i]).second) texr[i]->AddRef();
    g_draws.push_back(d);
  }
  LeaveCriticalSection(&g_cs);
  for (int i = 0; i < 4; ++i) {
    if (d.vb[i]) d.vb[i]->Release(); if (d.vscb[i]) d.vscb[i]->Release();
    if (pscb[i]) pscb[i]->Release(); if (texr[i]) texr[i]->Release();
  }
  if (d.ib) d.ib->Release();
}

static HRESULT STDMETHODCALLTYPE H_Map(ID3D11DeviceContext* c, ID3D11Resource* r, UINT sub, D3D11_MAP t, UINT f, D3D11_MAPPED_SUBRESOURCE* m) {
  DepthGuard g("Map");
  HRESULT hr = ((PFN_Map)OrigFor(c, 14))(RealOf(c), r, sub, t, f, m);
  UINT size;
  if (!g.deep && !g_optNoCb && SUCCEEDED(hr) && m && g_framesLeft.load() > 0 && (t == D3D11_MAP_WRITE_DISCARD || t == D3D11_MAP_WRITE_NO_OVERWRITE || t == D3D11_MAP_WRITE)
      && IsConstantBuffer(r, &size)) {
    // Dynamic vertex buffers: the engine rewrites the start of the per-draw instance stream before each draw
    // (WRITE_DISCARD), so copying the first 8 KB is enough and keeps the capture frame fast.
    D3D11_BUFFER_DESC bd; ((ID3D11Buffer*)r)->GetDesc(&bd);
    if (!(bd.BindFlags & D3D11_BIND_CONSTANT_BUFFER) && t == D3D11_MAP_WRITE_DISCARD && size > 8192) size = 8192;
    EnterCriticalSection(&g_cs); g_pendingMaps[r] = {m->pData, size}; LeaveCriticalSection(&g_cs);
  }
  return hr;
}
static void STDMETHODCALLTYPE H_Unmap(ID3D11DeviceContext* c, ID3D11Resource* r, UINT sub) {
  DepthGuard g("Unmap");
  if (!g.deep && !g_optNoCb && g_framesLeft.load() > 0) {
    EnterCriticalSection(&g_cs);
    auto it = g_pendingMaps.find(r);
    if (it != g_pendingMaps.end()) {
      std::string& sh = g_cbShadow[(ID3D11Buffer*)r];
      if (sh.size() < it->second.size) sh.resize(it->second.size);
      memcpy(&sh[0], it->second.ptr, it->second.size);
      g_pendingMaps.erase(it);
    }
    LeaveCriticalSection(&g_cs);
  }
  ((PFN_Unmap)OrigFor(c, 15))(RealOf(c), r, sub);
}
static void STDMETHODCALLTYPE H_UpdateSubresource(ID3D11DeviceContext* c, ID3D11Resource* r, UINT sub, const D3D11_BOX* box, const void* data, UINT rp, UINT dp) {
  DepthGuard g("UpdateSubresource");
  UINT size;
  if (!g.deep && !g_optNoCb && g_framesLeft.load() > 0 && data && IsConstantBuffer(r, &size)) {
    EnterCriticalSection(&g_cs);
    std::string& sh = g_cbShadow[(ID3D11Buffer*)r];
    if (!box) sh.assign((const char*)data, size);
    else { if (sh.size() < size) sh.resize(size); if (box->right <= size && box->left < box->right) memcpy(&sh[box->left], data, box->right - box->left); }
    LeaveCriticalSection(&g_cs);
  }
  ((PFN_UpdateSubresource)OrigFor(c, 48))(RealOf(c), r, sub, box, data, rp, dp);
}

typedef void (STDMETHODCALLTYPE* PFN_PSSetShaderResources)(ID3D11DeviceContext*, UINT, UINT, ID3D11ShaderResourceView* const*);
typedef void (STDMETHODCALLTYPE* PFN_PSSetShader)(ID3D11DeviceContext*, ID3D11PixelShader*, ID3D11ClassInstance* const*, UINT);
static void STDMETHODCALLTYPE H_PSSetShaderResources(ID3D11DeviceContext* c, UINT start, UINT n, ID3D11ShaderResourceView* const* v) {
  for (UINT k = start; k < start + n && k < 8; ++k) { g_texFresh |= 1u << k; g_texSinceShader |= 1u << k; }
  ((PFN_PSSetShaderResources)OrigFor(c, 8))(RealOf(c), start, n, v);
}
static void STDMETHODCALLTYPE H_PSSetShader(ID3D11DeviceContext* c, ID3D11PixelShader* ps, ID3D11ClassInstance* const* ci, UINT nci) {
  g_texSinceShader = 0;
  ((PFN_PSSetShader)OrigFor(c, 9))(RealOf(c), ps, ci, nci);
}
static void STDMETHODCALLTYPE H_DrawIndexed(ID3D11DeviceContext* c, UINT n, UINT si, INT bv) {
  DepthGuard g("DrawIndexed"); if (!g.deep) RecordDraw(RealOf(c), 0, n, 1, si, bv, 0, 0); ((PFN_DrawIndexed)OrigFor(c, 12))(RealOf(c), n, si, bv);
}
static void STDMETHODCALLTYPE H_Draw(ID3D11DeviceContext* c, UINT n, UINT sv) {
  DepthGuard g("Draw"); if (!g.deep) RecordDraw(RealOf(c), 1, n, 1, 0, 0, sv, 0); ((PFN_Draw)OrigFor(c, 13))(RealOf(c), n, sv);
}
static void STDMETHODCALLTYPE H_DrawIndexedInstanced(ID3D11DeviceContext* c, UINT n, UINT inst, UINT si, INT bv, UINT sinst) {
  DepthGuard g("DrawIndexedInstanced"); if (!g.deep) RecordDraw(RealOf(c), 2, n, inst, si, bv, 0, sinst); ((PFN_DrawIndexedInstanced)OrigFor(c, 20))(RealOf(c), n, inst, si, bv, sinst);
}
static void STDMETHODCALLTYPE H_DrawInstanced(ID3D11DeviceContext* c, UINT n, UINT inst, UINT sv, UINT sinst) {
  DepthGuard g("DrawInstanced"); if (!g.deep) RecordDraw(RealOf(c), 3, n, inst, 0, 0, sv, sinst); ((PFN_DrawInstanced)OrigFor(c, 21))(RealOf(c), n, inst, sv, sinst);
}

// ------------------------------------------------------------------ readback + output (render thread)
static bool WriteFileAll(const std::string& path, const void* data, size_t n) {
  FILE* f = fopen(path.c_str(), "wb"); if (!f) return false;
  fwrite(data, 1, n, f); fclose(f); return true;
}

struct Nl3dBuf { bool isVB; uintptr_t obj; uint16_t stride; uint32_t count; uint8_t types[32]; uint16_t offs[32]; };

// Scan committed private memory for engine VertexBuffer/IndexBuffer objects (by vtable) to learn vertex layouts.
static void ScanNl3dBuffers(std::map<ID3D11Buffer*, Nl3dBuf>& out) {
  // ReadProcessMemory on ourselves: a page freed by another thread mid-scan gives an error instead of a crash.
  HANDLE self = GetCurrentProcess();
  const size_t CH = 1 << 20, OV = 0x100;
  std::vector<uint8_t> buf(CH + OV);
  MEMORY_BASIC_INFORMATION mbi; uint8_t* p = nullptr;
  while (VirtualQuery(p, &mbi, sizeof mbi)) {
    bool ok = mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE && (mbi.Protect & PAGE_READWRITE) && !(mbi.Protect & PAGE_GUARD);
    if (ok) {
      uint8_t* base = (uint8_t*)mbi.BaseAddress; size_t n = mbi.RegionSize;
      for (size_t c = 0; c < n; c += CH) {
        size_t want = std::min(CH + OV, n - c); SIZE_T got = 0;
        if (!ReadProcessMemory(self, base + c, buf.data(), want, &got) || got < 0xc0) continue;
        size_t lim = std::min(CH, got);
        for (size_t o = 0; o + 0xc0 <= got && o < lim; o += 8) {
          uint8_t* b = buf.data() + o;
          uintptr_t v = *(uintptr_t*)b;
          if (v != VT_D3D11_VERTEXBUFFER && v != VT_D3D11_INDEXBUFFER) continue;
          Nl3dBuf nb{}; nb.obj = (uintptr_t)(base + c + o);
          ID3D11Buffer* bufp;
          if (v == VT_D3D11_VERTEXBUFFER) {
            nb.isVB = true; memcpy(nb.types, b + 0x08, 32);
            nb.stride = *(uint16_t*)(b + 0x2e); nb.count = *(uint32_t*)(b + 0x30);
            memcpy(nb.offs, b + 0x40, 64); bufp = *(ID3D11Buffer**)(b + 0xb8);
          } else {
            nb.isVB = false; nb.count = *(uint32_t*)(b + 0x10); bufp = *(ID3D11Buffer**)(b + 0x50);
          }
          if (bufp) out[bufp] = nb;
        }
      }
    }
    p = (uint8_t*)mbi.BaseAddress + mbi.RegionSize;
    if ((uintptr_t)p >= 0x7FFFFFFF0000ull) break;
  }
}

static void FinishCapture() {
  ID3D11Device* dev = *(ID3D11Device**)G_D3D_DEVICE;
  if (!dev) { g_result = "ERR no device"; return; }
  dev->AddRef();
  ID3D11DeviceContext* ctx = nullptr; dev->GetImmediateContext(&ctx);
  Log("finish: %zu draws, %zu buffers, %zu cb blobs - reading back", g_draws.size(), g_buffers.size(), g_cbBlobs.size());
  CreateDirectoryA(g_outDir.c_str(), nullptr);
  std::string bufDir = g_outDir + "\\buffers"; CreateDirectoryA(bufDir.c_str(), nullptr);

  EnterCriticalSection(&g_cs);
  std::vector<DrawRec> draws; draws.swap(g_draws);
  std::vector<std::string> cbBlobs; cbBlobs.swap(g_cbBlobs); g_cbBlobIndex.clear(); g_cbShadow.clear(); g_pendingMaps.clear();
  std::set<ID3D11Buffer*> bufs; bufs.swap(g_buffers);
  std::set<ID3D11Resource*> texs; texs.swap(g_textures);
  g_seen.clear();
  LeaveCriticalSection(&g_cs);

  size_t bytes = 0; int nOk = 0;
  std::string bj = "{";
  bool firstB = true;
  for (ID3D11Buffer* b : bufs) {
    D3D11_BUFFER_DESC desc; b->GetDesc(&desc);
    D3D11_BUFFER_DESC sd = {}; sd.ByteWidth = desc.ByteWidth; sd.Usage = D3D11_USAGE_STAGING; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* st = nullptr;
    char name[40]; snprintf(name, sizeof name, "%016llx", (unsigned long long)(uintptr_t)b);
    if (SUCCEEDED(dev->CreateBuffer(&sd, nullptr, &st))) {
      ctx->CopyResource(st, b);
      D3D11_MAPPED_SUBRESOURCE m;
      if (SUCCEEDED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
        WriteFileAll(bufDir + "\\" + name + ".bin", m.pData, desc.ByteWidth);
        ctx->Unmap(st, 0); bytes += desc.ByteWidth; nOk++;
      }
      st->Release();
    }
    char e[200];
    snprintf(e, sizeof e, "%s\"%s\":{\"bytes\":%u,\"bind\":%u,\"usage\":%u}", firstB ? "" : ",", name, desc.ByteWidth, desc.BindFlags, desc.Usage);
    bj += e; firstB = false;
    b->Release();
  }
  bj += "}";

  // textures: top mip of every 2D texture a captured draw sampled, saved as .dds (DX10 header)
  std::string texDir = g_outDir + "\\textures"; CreateDirectoryA(texDir.c_str(), nullptr);
  std::string tj = "{"; bool firstT = true; int nTex = 0; size_t texBytes = 0;
  for (ID3D11Resource* r : texs) {
    D3D11_RESOURCE_DIMENSION dim; r->GetType(&dim);
    char name[40]; snprintf(name, sizeof name, "%016llx", (unsigned long long)(uintptr_t)r);
    if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
      D3D11_TEXTURE2D_DESC td; ((ID3D11Texture2D*)r)->GetDesc(&td);
      bool ok = td.SampleDesc.Count == 1 && !(td.BindFlags & D3D11_BIND_DEPTH_STENCIL) && (size_t)td.Width * td.Height <= (size_t)8192 * 8192;
      if (ok) {
        D3D11_TEXTURE2D_DESC sd = td; sd.MipLevels = 1; sd.ArraySize = 1; sd.Usage = D3D11_USAGE_STAGING;
        sd.BindFlags = 0; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ; sd.MiscFlags = 0;
        ID3D11Texture2D* st = nullptr;
        if (SUCCEEDED(dev->CreateTexture2D(&sd, nullptr, &st))) {
          ctx->CopySubresourceRegion(st, 0, 0, 0, 0, r, 0, nullptr);
          D3D11_MAPPED_SUBRESOURCE m;
          if (SUCCEEDED(ctx->Map(st, 0, D3D11_MAP_READ, 0, &m))) {
            // rows: block-compressed formats store 4 pixel rows per row of blocks
            DXGI_FORMAT f = td.Format;
            bool bc = (f >= DXGI_FORMAT_BC1_TYPELESS && f <= DXGI_FORMAT_BC5_SNORM) || (f >= DXGI_FORMAT_BC6H_TYPELESS && f <= DXGI_FORMAT_BC7_UNORM_SRGB);
            UINT rows = bc ? (td.Height + 3) / 4 : td.Height;
            std::string file;
            uint32_t hdr[32] = {};                      // "DDS " + DDS_HEADER (31 dwords)
            hdr[0] = 0x20534444; hdr[1] = 124; hdr[2] = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000;
            hdr[3] = td.Height; hdr[4] = td.Width; hdr[5] = m.RowPitch * rows; hdr[7] = 1;
            hdr[19] = 32; hdr[20] = 0x4; hdr[21] = 0x30315844;   // ddpf: FOURCC 'DX10'
            hdr[27] = 0x1000;                           // caps: texture
            uint32_t dx10[5] = {(uint32_t)f, 3 /*TEXTURE2D*/, 0, 1, 0};
            file.append((const char*)hdr, sizeof hdr); file.append((const char*)dx10, sizeof dx10);
            file.append((const char*)m.pData, (size_t)m.RowPitch * rows);
            ctx->Unmap(st, 0);
            WriteFileAll(texDir + "\\" + name + ".dds", file.data(), file.size());
            texBytes += file.size(); nTex++;
            char e[260];
            snprintf(e, sizeof e, "%s\"%s\":{\"w\":%u,\"h\":%u,\"format\":%d,\"pitch\":%u,\"bind\":%u,\"mips\":%u}\n",
                     firstT ? "" : ",", name, td.Width, td.Height, (int)f, m.RowPitch, td.BindFlags, td.MipLevels);
            tj += e; firstT = false;
          }
          st->Release();
        }
      }
    }
    r->Release();
  }
  tj += "}";
  WriteFileAll(g_outDir + "\\textures.json", tj.data(), tj.size());
  Log("finish: %d textures (%zu bytes)", nTex, texBytes);

  std::string dj = "[";
  for (size_t i = 0; i < draws.size(); ++i) {
    const DrawRec& d = draws[i];
    char e[1900];
    snprintf(e, sizeof e,
      "%s{\"kind\":%d,\"count\":%u,\"instances\":%u,\"startIndex\":%u,\"baseVertex\":%d,\"startVertex\":%u,\"startInstance\":%u,"
      "\"vb\":[\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\"],\"stride\":[%u,%u,%u,%u],\"offset\":[%u,%u,%u,%u],"
      "\"ib\":\"%016llx\",\"ibFormat\":%d,\"ibOffset\":%u,\"topology\":%d,\"layout\":\"%p\",\"vs\":\"%p\",\"ps\":\"%p\",\"tex0\":\"%p\","
      "\"vscb\":[\"%p\",\"%p\",\"%p\",\"%p\"],\"cb\":[%d,%d,%d,%d],\"inst\":%d,\"frame\":%d,"
      "\"tex\":[\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\"],\"pscb\":[%d,%d,%d,%d],\"texSet\":%d}\n",
      i ? "," : "", d.kind, d.count, d.instances, d.startIndex, d.baseVertex, d.startVertex, d.startInstance,
      (unsigned long long)(uintptr_t)d.vb[0], (unsigned long long)(uintptr_t)d.vb[1], (unsigned long long)(uintptr_t)d.vb[2], (unsigned long long)(uintptr_t)d.vb[3],
      d.stride[0], d.stride[1], d.stride[2], d.stride[3], d.offset[0], d.offset[1], d.offset[2], d.offset[3],
      (unsigned long long)(uintptr_t)d.ib, (int)d.ibFmt, d.ibOffset, (int)d.topo, d.layout, d.vs, d.ps, d.tex0,
      (void*)d.vscb[0], (void*)d.vscb[1], (void*)d.vscb[2], (void*)d.vscb[3], d.cbBlob[0], d.cbBlob[1], d.cbBlob[2], d.cbBlob[3], d.instBlob, d.frame,
      (unsigned long long)(uintptr_t)d.tex[0], (unsigned long long)(uintptr_t)d.tex[1], (unsigned long long)(uintptr_t)d.tex[2], (unsigned long long)(uintptr_t)d.tex[3],
      d.psBlob[0], d.psBlob[1], d.psBlob[2], d.psBlob[3], d.texSet);
    dj += e;
  }
  dj += "]";
  Log("finish: readback done (%d buffers, %zu bytes), writing json", nOk, bytes);
  WriteFileAll(g_outDir + "\\draws.json", dj.data(), dj.size());
  WriteFileAll(g_outDir + "\\buffers.json", bj.data(), bj.size());
  { // constant buffer blobs: u32 count, then per blob u32 size + bytes
    std::string cb; uint32_t n = (uint32_t)cbBlobs.size(); cb.append((const char*)&n, 4);
    for (auto& b : cbBlobs) { uint32_t z = (uint32_t)b.size(); cb.append((const char*)&z, 4); cb += b; }
    WriteFileAll(g_outDir + "\\cbuffers.bin", cb.data(), cb.size());
  }

  // engine buffer layouts
  std::map<ID3D11Buffer*, Nl3dBuf> nl3d;
  if (!g_optNoScan) { Log("finish: scanning heap for engine buffers"); ScanNl3dBuffers(nl3d); }
  std::string nj = "{"; bool first = true;
  for (auto& kv : nl3d) {
    const Nl3dBuf& nb = kv.second;
    char e[900]; int k = snprintf(e, sizeof e, "%s\"%016llx\":{\"vb\":%d,\"obj\":\"%llx\",\"stride\":%u,\"count\":%u,\"types\":[",
                                  first ? "" : ",", (unsigned long long)(uintptr_t)kv.first, nb.isVB, (unsigned long long)nb.obj, nb.stride, nb.count);
    for (int i = 0; i < 32 && nb.types[i]; ++i) k += snprintf(e + k, sizeof e - k, "%s%u", i ? "," : "", nb.types[i]);
    k += snprintf(e + k, sizeof e - k, "],\"offsets\":[");
    for (int i = 0; i < 32 && nb.types[i]; ++i) k += snprintf(e + k, sizeof e - k, "%s%u", i ? "," : "", nb.offs[i]);
    snprintf(e + k, sizeof e - k, "]}\n");
    nj += e; first = false;
  }
  nj += "}";
  WriteFileAll(g_outDir + "\\nl3d_buffers.json", nj.data(), nj.size());

  char r[256];
  snprintf(r, sizeof r, "OK draws=%zu buffers=%d bytes=%zu textures=%d nl3dBuffers=%zu dir=%s", draws.size(), nOk, bytes, nTex, nl3d.size(), g_outDir.c_str());
  g_result = r;
  Log("%s", r);
  ctx->Release(); dev->Release();
}

// ------------------------------------------------------------------ full-scene override (render thread only)
typedef void (*PFN_NLMainCull)(void*, void*, uint8_t*, void*, uint64_t);
static PFN_NLMainCull o_NLMainCull;
static std::atomic<bool> g_fullScene{false};   // set by "CAPTURE ... full", cleared when the capture ends
static bool g_fullApplied = false, g_sobPatchable = false;
static uint8_t g_savedOccMode = 0;
static int g_fullFrames = 0;

static void WriteCode(uintptr_t at, const uint8_t* bytes, size_t n) {
  DWORD old; VirtualProtect((void*)at, n, PAGE_EXECUTE_READWRITE, &old);
  memcpy((void*)at, bytes, n);
  VirtualProtect((void*)at, n, old, &old);
  FlushInstructionCache(GetCurrentProcess(), (void*)at, n);
}

static void SetFullOverrides(bool on) {
  if (on == g_fullApplied) return;
  static const uint8_t retTrue[3] = {0xB0, 0x01, 0xC3};
  if (on) {
    g_savedOccMode = *(volatile uint8_t*)G_OCCLUSION_MODE;
    if (g_savedOccMode != 0) *(volatile uint8_t*)G_OCCLUSION_MODE = 2;
    if (g_sobPatchable) for (uintptr_t f : FN_SOB_TESTS) WriteCode(f, retTrue, 3);
    g_fullFrames = 0;
  } else {
    if (*(volatile uint8_t*)G_OCCLUSION_MODE == 2) *(volatile uint8_t*)G_OCCLUSION_MODE = g_savedOccMode;
    if (g_sobPatchable) for (uintptr_t f : FN_SOB_TESTS) WriteCode(f, SOB_PROLOGUE, 3);
  }
  g_fullApplied = on;
  Log("full-scene overrides %s (occlusion mode %d, %d frame(s))", on ? "ON" : "OFF", g_savedOccMode, g_fullFrames);
}

// Runs on the game's main thread right before the scene cull; the cull worker is idle at this point.
static void H_NLMainCull(void* self, void* renderer, uint8_t* view, void* p4, uint64_t p5) {
  bool want = g_fullScene.load();
  SetFullOverrides(want);
  if (want && view) {
    *(float*)(view + 0x14) = 0.0f;       // LOD offset
    *(float*)(view + 0x18) = 1e-6f;      // LOD scale: every object picks its finest LOD and none is faded out
    *(uint32_t*)(view + 0x3ec) = 0;      // no frustum planes
    g_fullFrames++;
  }
  o_NLMainCull(self, renderer, view, p4, p5);
}

static void InstallCullHook() {
  g_sobPatchable = memcmp((void*)FN_SOB_TESTS[0], SOB_PROLOGUE, 3) == 0 && memcmp((void*)FN_SOB_TESTS[1], SOB_PROLOGUE, 3) == 0;
  void** vt = (void**)VT_NLMAIN;
  void* cur = vt[NLMAIN_CULL_SLOT];
  if ((uintptr_t)cur != 0x14029bf00) Log("NLMain cull slot is %p (expected 14029bf00) - chaining anyway", cur);
  o_NLMainCull = (PFN_NLMainCull)PatchVtable(vt, NLMAIN_CULL_SLOT, (void*)&H_NLMainCull);
  Log("cull hook installed (sob tests %s)", g_sobPatchable ? "patchable" : "NOT patchable - left alone");
}

static void* g_presentTramp = nullptr;
typedef uint64_t (*PFN_EnginePresent)(void*, void*, void*, void*);

static void BuildWrapper() {
  uint8_t* code = (uint8_t*)VirtualAlloc(nullptr, WRAP_SLOTS * 16, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
  static void* vt[WRAP_SLOTS];
  for (int i = 0; i < WRAP_SLOTS; ++i) {
    uint8_t* t = code + i * 16;
    // mov rcx,[rcx+8] ; mov rax,[rcx] ; jmp [rax+disp32]
    const uint8_t stub[] = {0x48, 0x8B, 0x49, 0x08, 0x48, 0x8B, 0x01, 0xFF, 0xA0};
    memcpy(t, stub, sizeof stub);
    int32_t disp = i * 8; memcpy(t + 9, &disp, 4);
    t[13] = 0xCC; t[14] = 0xCC; t[15] = 0xCC;
    vt[i] = t;
  }
  vt[12] = (void*)&H_DrawIndexed; vt[13] = (void*)&H_Draw; vt[14] = (void*)&H_Map; vt[15] = (void*)&H_Unmap;
  vt[20] = (void*)&H_DrawIndexedInstanced; vt[21] = (void*)&H_DrawInstanced; vt[48] = (void*)&H_UpdateSubresource;
  vt[8] = (void*)&H_PSSetShaderResources; vt[9] = (void*)&H_PSSetShader;
  FlushInstructionCache(GetCurrentProcess(), code, WRAP_SLOTS * 16);
  g_wrap.vtbl = vt;
}

// Called on the game's render thread at the start of every engine present (all draws of the frame are done).
static void OnEngineFrame() {
  ID3D11DeviceContext** slot = (ID3D11DeviceContext**)G_D3D_CONTEXT;
  bool wrapped = *slot == (ID3D11DeviceContext*)&g_wrap;
  if (g_framesLeft.load() > 0) {
    if (!wrapped) {                         // start: swap our wrapper in for the next frame(s)
      if (g_fullScene.load() && g_fullFrames < 2) return;   // let the cull/LOD overrides take effect first
      g_wrap.real = *slot; *slot = (ID3D11DeviceContext*)&g_wrap;
      o_DrawIndexed = (PFN_DrawIndexed)1;   // marks "capturing"
      Log("wrapper in (real context %p)", g_wrap.real);
      return;
    }
    g_frameIndex++;
    if (g_framesLeft.fetch_sub(1) == 1) {
      *slot = g_wrap.real;                  // restore before reading back
      Log("wrapper out after %d frame(s)", g_frameIndex);
      FinishCapture();
      g_fullScene = false;
      SetEvent(g_doneEvent);
    }
  } else if (wrapped) {
    *slot = g_wrap.real;                    // safety
  }
}

static uint64_t H_EnginePresent(void* a, void* b, void* c, void* d) {
  OnEngineFrame();
  return ((PFN_EnginePresent)g_presentTramp)(a, b, c, d);
}

static bool SupportedBuild() {
  auto base = (BYTE*)GetModuleHandleA(nullptr);
  auto nt = (IMAGE_NT_HEADERS*)(base + ((IMAGE_DOS_HEADER*)base)->e_lfanew);
  return (uintptr_t)base == 0x140000000 && nt->FileHeader.TimeDateStamp == 0x696F4197;
}

static bool InstallHooks() {
  if (!SupportedBuild()) { Log("unsupported game build - nothing hooked"); return false; }
  uint8_t* target = (uint8_t*)FN_ENGINE_PRESENT;
  bool chained = target[0] == 0xFF && target[1] == 0x25 && *(uint32_t*)(target + 2) == 0 && target[14] == 0x90;  // an older copy of us already hooked it
  if (!chained && memcmp(target, ENGINE_PRESENT_PROLOGUE, 15) != 0) { Log("engine present prologue mismatch - wrong game build?"); return false; }
  if (chained) Log("engine present already hooked by an earlier NL2ModelExport - chaining");
  BuildWrapper();
  uint8_t* t = (uint8_t*)VirtualAlloc(nullptr, 64, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
  memcpy(t, target, 15);
  t[15] = 0xFF; t[16] = 0x25; memset(t + 17, 0, 4);
  uint64_t back = (uint64_t)(target + 15); memcpy(t + 21, &back, 8);
  g_presentTramp = t;
  uint8_t patch[15] = {0xFF, 0x25, 0, 0, 0, 0}; uint64_t dst = (uint64_t)&H_EnginePresent; memcpy(patch + 6, &dst, 8); patch[14] = 0x90;
  DWORD old; VirtualProtect(target, 15, PAGE_EXECUTE_READWRITE, &old); memcpy(target, patch, 15); VirtualProtect(target, 15, old, &old);
  FlushInstructionCache(GetCurrentProcess(), target, 15);
  Log("engine present hooked (%p), wrapper context ready (%d slots)", target, WRAP_SLOTS);
  InstallCullHook();
  return true;
}

// ------------------------------------------------------------------ tiny text server
static std::string g_portFile;
static DWORD WINAPI Server(LPVOID) {
  WSADATA wd; WSAStartup(MAKEWORD(2, 2), &wd);
  SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  int port = 0;
  for (int p = 15153; p < 15163 && !port; ++p) {
    sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_port = htons((u_short)p); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(ls, (sockaddr*)&a, sizeof a) == 0 && listen(ls, 2) == 0) port = p;
  }
  if (!port) { Log("no free port 15153-15162"); return 0; }
  Log("listening on 127.0.0.1:%d", port);
  FILE* pf = fopen(g_portFile.c_str(), "w"); if (pf) { fprintf(pf, "%d", port); fclose(pf); }
  for (;;) {
    SOCKET c = accept(ls, nullptr, nullptr); if (c == INVALID_SOCKET) break;
    char buf[1024]; int n = recv(c, buf, sizeof buf - 1, 0);
    if (n > 0) {
      buf[n] = 0; while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
      std::string reply;
      if (!strncmp(buf, "PING", 4)) reply = "OK NL2ModelExport 0.11";
      else if (!strncmp(buf, "CAPTURE ", 8)) {
        // CAPTURE <frames> [nocb] [noscan] [full] [inst] <dir>
        int frames = atoi(buf + 8); const char* dir = strchr(buf + 8, ' ');
        g_optNoCb = g_optNoScan = g_optInstOnly = false; bool full = false;
        while (dir) {
          if (!strncmp(dir, " nocb ", 6)) { g_optNoCb = true; dir += 5; continue; }
          if (!strncmp(dir, " noscan ", 8)) { g_optNoScan = true; dir += 7; continue; }
          if (!strncmp(dir, " full ", 6)) { full = true; dir += 5; continue; }
          if (!strncmp(dir, " inst ", 6)) { g_optInstOnly = true; dir += 5; continue; }
          break;
        }
        if (frames <= 0 || !dir) reply = "ERR usage: CAPTURE <frames> <dir>";
        else if (g_framesLeft.load() > 0) reply = "ERR capture already running";
        else {
          g_outDir = dir + 1; g_result.clear(); ResetEvent(g_doneEvent);
          EnterCriticalSection(&g_cs); g_draws.clear(); g_seen.clear(); LeaveCriticalSection(&g_cs);
          g_frameIndex = 0; g_fullScene = full; g_framesLeft = frames;
          Log("capture armed: %d frames -> %s (cb shadow %s, heap scan %s, full scene %s)", frames, g_outDir.c_str(), g_optNoCb ? "off" : "on", g_optNoScan ? "off" : "on", full ? "on" : "off");
          if (WaitForSingleObject(g_doneEvent, 300000) == WAIT_OBJECT_0) reply = g_result;
          else { g_framesLeft = 0; g_fullScene = false; reply = "ERR timeout (is the game rendering? not minimised?)"; }
        }
      } else reply = "ERR unknown command";
      reply += "\n"; send(c, reply.c_str(), (int)reply.size(), 0);
    }
    closesocket(c);
  }
  return 0;
}

static DWORD WINAPI Init(LPVOID self) {
  char path[MAX_PATH]; GetModuleFileNameA((HMODULE)self, path, MAX_PATH);
  char* dot = strrchr(path, '.'); if (dot) *dot = 0;
  g_portFile = std::string(path) + ".port";
  strcat(path, ".log");
  InitializeCriticalSection(&g_logCs); InitializeCriticalSection(&g_cs);
  g_log = fopen(path, "a");
  g_doneEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
  Log("NL2ModelExport starting");
  if (!InstallHooks()) { Log("hook install failed"); return 0; }
  Server(nullptr);
  return 0;
}

BOOL WINAPI DllMain(HINSTANCE h, DWORD r, LPVOID) {
  if (r == DLL_PROCESS_ATTACH) { DisableThreadLibraryCalls(h); CloseHandle(CreateThread(nullptr, 0, Init, h, 0, nullptr)); }
  return TRUE;
}
