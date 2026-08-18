; Inno Setup script for PIV Suite (CPU backend).
; Wraps the PyInstaller --onedir bundle at dist\PIV_Suite\ into a
; standard Windows installer with Start Menu / optional Desktop
; shortcuts and a proper uninstaller.
;
; Build order:
;   1. pyinstaller ... src/piv_suite_gui/app.py   (produces dist\PIV_Suite\)
;   2. ISCC.exe installer\piv_suite.iss           (produces Output\*.exe)
;
; GPU backend (cupy + openpiv-python-gpu) is NOT bundled -- it requires a
; matching CUDA toolkit/driver on the target machine and is set up
; separately per INSTALL_WINDOWS.md, same as the dev workflow.

#define MyAppName "PIV Suite"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "SPC3MN"
#define MyAppExeName "PIV_Suite.exe"

[Setup]
AppId={{B6E1F5B0-6C0B-4A9A-9C4A-6E7F9E1C6B10}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Per-user install (like VS Code) -- no admin/UAC elevation needed.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=PIV_Suite_Setup_{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "dist\PIV_Suite\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
