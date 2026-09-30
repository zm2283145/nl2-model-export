@echo off
rem Builds the tools into build\ with MinGW-w64 (default location C:\msys64\mingw64\bin).
rem   build.cmd          -> build\NL2ModelExport.dll, build\nl2vfsdump.dll, build\nl2inject.exe
rem   build.cmd 12       -> build\NL2ModelExport_v12.dll (a DLL that is loaded in the game can't be overwritten,
rem                         so bump the number when rebuilding while the game is running)
if not defined MINGW set MINGW=C:\msys64\mingw64\bin
set PATH=%MINGW%;%PATH%
cd /d %~dp0
if not exist build mkdir build
set OUT=build\NL2ModelExport.dll
if not "%1"=="" set OUT=build\NL2ModelExport_v%1.dll
g++ -O2 -std=c++17 -shared -static -static-libgcc -static-libstdc++ -o %OUT% src\nl2export.cpp -ld3d11 -ldxgi -lws2_32 -ldbghelp
if errorlevel 1 exit /b 1
g++ -O2 -std=c++17 -shared -static -static-libgcc -static-libstdc++ -o build\nl2vfsdump.dll src\nl2vfsdump.cpp
if errorlevel 1 exit /b 1
g++ -O2 -std=c++17 -municode -static -static-libgcc -static-libstdc++ -o build\nl2inject.exe src\inject.cpp
if errorlevel 1 exit /b 1
echo built %OUT%, build\nl2vfsdump.dll, build\nl2inject.exe
