// nl2vfsdump: injected helper that reads files through NoLimits 2's own virtual file system
// (packages included) and writes them to disk. Reads <dll>.txt (one "base:data/..." path per line),
// writes into <dll dir>\vfs\<path>, logs to <dll>.log, then unloads itself.
// Target: nolimits2stm.exe Steam build, PE timestamp 0x696F4197.
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
// MSVC std::string layout used by the game.
struct MsvcString {
  union { char buf[16]; char* ptr; };
  size_t size, cap;
};
using OpenFn = void** (*)(void** out, MsvcString* path);
const uintptr_t FN_VFS_OPEN = 0x14058cfd0;
// nl3d::File vtable slots
const int VT_SIZE = 0x30 / 8, VT_OPEN = 0x40 / 8, VT_READ = 0x48 / 8, VT_CLOSE = 0x78 / 8, VT_RELEASE = 0x18 / 8;

FILE* g_log;
void Log(const char* f, ...) {
  va_list a; va_start(a, f); vfprintf(g_log, f, a); va_end(a); fputc('\n', g_log); fflush(g_log);
}

bool CheckExe() {
  auto base = (BYTE*)GetModuleHandleA(nullptr);
  auto nt = (IMAGE_NT_HEADERS*)(base + ((IMAGE_DOS_HEADER*)base)->e_lfanew);
  return (uintptr_t)base == 0x140000000 && nt->FileHeader.TimeDateStamp == 0x696F4197;
}

void MakeDirs(std::string p) {
  for (size_t i = 3; i < p.size(); ++i)
    if (p[i] == '\\' || p[i] == '/') { char c = p[i]; p[i] = 0; CreateDirectoryA(p.c_str(), nullptr); p[i] = c; }
}

long long Dump(const std::string& vpath, const std::string& outPath) {
  std::vector<char> s(vpath.begin(), vpath.end()); s.push_back(0);
  MsvcString ms = {}; ms.size = vpath.size();
  if (vpath.size() < 16) { memcpy(ms.buf, s.data(), s.size()); ms.cap = 15; }
  else { ms.ptr = s.data(); ms.cap = vpath.size(); }
  void* f = nullptr;
  ((OpenFn)FN_VFS_OPEN)(&f, &ms);
  if (!f) return -1;
  void** vt = *(void***)f;
  long long result = -2;
  if (((bool (*)(void*))vt[VT_OPEN])(f)) {
    long long n = ((long long (*)(void*))vt[VT_SIZE])(f);
    std::vector<char> data(n > 0 ? (size_t)n : 1);
    size_t got = n > 0 ? ((size_t (*)(void*, void*, size_t, size_t))vt[VT_READ])(f, data.data(), 1, (size_t)n) : 0;
    ((bool (*)(void*))vt[VT_CLOSE])(f);
    MakeDirs(outPath);
    if (FILE* o = fopen(outPath.c_str(), "wb")) { fwrite(data.data(), 1, got, o); fclose(o); }
    result = (long long)got;
  }
  ((void (*)(void*))vt[VT_RELEASE])(f);
  return result;
}

DWORD WINAPI Run(LPVOID self) {
  char path[MAX_PATH]; GetModuleFileNameA((HMODULE)self, path, MAX_PATH);
  std::string base = path; base = base.substr(0, base.rfind('.'));
  std::string dir = base.substr(0, base.rfind('\\') + 1) + "vfs\\";
  g_log = fopen((base + ".log").c_str(), "a");
  if (!g_log) FreeLibraryAndExitThread((HMODULE)self, 0);
  if (!CheckExe()) { Log("unsupported executable"); fclose(g_log); FreeLibraryAndExitThread((HMODULE)self, 0); }
  FILE* l = fopen((base + ".txt").c_str(), "r");
  int ok = 0, bad = 0;
  if (l) {
    char line[1024];
    while (fgets(line, sizeof line, l)) {
      std::string v = line;
      while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' ')) v.pop_back();
      if (v.empty()) continue;
      std::string rel = v; size_t c = rel.find(':'); if (c != std::string::npos) rel = rel.substr(c + 1);
      for (auto& ch : rel) if (ch == '/') ch = '\\';
      long long n = Dump(v, dir + rel);
      if (n >= 0) ++ok; else ++bad;
      if (n >= 0) Log("%s %lld", v.c_str(), n);
    }
    fclose(l);
  }
  Log("done: %d ok, %d failed", ok, bad);
  fclose(g_log);
  FreeLibraryAndExitThread((HMODULE)self, 0);
}
}  // namespace

BOOL WINAPI DllMain(HINSTANCE h, DWORD r, LPVOID) {
  if (r == DLL_PROCESS_ATTACH) { DisableThreadLibraryCalls(h); CloseHandle(CreateThread(nullptr, 0, Run, h, 0, nullptr)); }
  return TRUE;
}
