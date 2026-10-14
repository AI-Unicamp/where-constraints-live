@echo off
rem Build the 3D lattice search (MSVC). Adjust the vcvars64.bat path to your Visual Studio installation.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cl /nologo /std:c++17 /O2 /EHsc /Fe:"%~dp0zhou3d.exe" "%~dp0zhou3d.cpp"
