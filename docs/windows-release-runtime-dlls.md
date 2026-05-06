# Windows release runtime DLLs

Testing the `v0.1.0` `libtotalseg-windows-x64.zip` asset showed that `totalseg_cli.exe` exits with `STATUS_DLL_NOT_FOUND` (`-1073741515`) when started from the extracted package.

The release package currently contains only:

- `bin/totalseg_cli.exe`
- `lib/totalseg.dll`
- `lib/totalseg.lib`

The direct PE imports show these runtime DLLs are required for a runnable Windows package:

- `totalseg.dll` next to `totalseg_cli.exe`, or `lib/` added to `PATH`
- `onnxruntime.dll`
- `onnxruntime_providers_shared.dll`
- zlib runtime DLL (`zlib1.dll` in the existing release build; `z.dll` when rebuilt with the current vcpkg package)
- `MSVCP140.dll`
- `MSVCP140_1.dll`
- `VCRUNTIME140.dll`
- `VCRUNTIME140_1.dll`
- UCRT API-set DLLs, normally provided by the system or Visual C++ runtime installation

The CMake build uses `$<TARGET_RUNTIME_DLLS:totalseg_cli>` to copy vcpkg-provided runtime DLLs, and `InstallRequiredSystemLibraries` to copy the MSVC runtime DLLs when needed. The GitHub Actions packaging step should use `cmake --install` and package the resulting `dist/` directory.

After rebuilding locally with CMake, Visual Studio Build Tools, vcpkg, and ONNX Runtime 1.22.0, the fixed package layout starts successfully with:

```powershell
bin\totalseg_cli.exe --help
```
