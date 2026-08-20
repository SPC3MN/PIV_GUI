; Inno Setup script for PIV Suite.
; Wraps the PyInstaller --onedir bundle at dist\PIV_Suite\ into a
; standard Windows installer with Start Menu / optional Desktop
; shortcuts and a proper uninstaller.
;
; Build order (or just run build_installer.ps1, which does all three):
;   1. prepare_gpu_assets.ps1                     (produces assets\python-embed\)
;   2. pyinstaller ... src/piv_suite_gui/app.py    (produces dist\PIV_Suite\)
;   3. ISCC.exe installer\piv_suite.iss            (produces Output\*.exe)
;
; The CPU/GUI app is bundled as-is in [Files]. GPU support (cupy +
; openpiv-python-gpu) is NOT baked into this installer file -- the wizard
; asks which CUDA version (if any) the user has, and if not "CPU only",
; DOWNLOADS the matching small cupy wheel + cuda-pathfinder (via a
; bundled pip-capable Python, see prepare_gpu_assets.ps1) plus
; openpiv-python-gpu's source, at INSTALL time, into {app}\_internal
; (confirmed on real hardware: PyInstaller's onedir bundle looks for
; importable packages there, same as its own bundled scipy/numpy).
;
; Why download instead of bundle: cupy's own wheel is small (~35MB, no
; CUDA runtime inside it -- it locates an EXISTING NVIDIA CUDA Toolkit on
; the machine via the cuda-pathfinder package) but pre-baking all three
; CUDA variants into the installer file would still balloon it, and the
; wheel must match cp313-win_amd64 exactly, which pip resolves reliably
; and a hand-rolled Pascal-Script PyPI-API parser would not. This means
; GPU setup needs internet access at INSTALL time (not just build time),
; and the target machine still needs the actual NVIDIA CUDA Toolkit
; installed separately -- this installer cannot provide that (multi-GB,
; its own separate license/installer). The wizard page says so.

#define MyAppName "PIV Suite"
#define MyAppVersion "0.2.3"
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
; Install-time-only tool for fetching GPU packages -- lives in {tmp}, not
; copied into {app}, so it's not part of the persistent install. {tmp}
; destinations are extracted automatically before [Code] runs, so no
; per-file "dontcopy" bookkeeping is needed (the ~30MB extracts every
; run regardless of the GPU choice -- an acceptable tradeoff for not
; having to enumerate every embeddable-Python file individually).
Source: "assets\python-embed\*"; DestDir: "{tmp}\pyembed"; Flags: recursesubdirs
Source: "gpu_setup\fetch_openpiv_gpu.py"; DestDir: "{tmp}\pyembed"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; gpu_setup_log.txt is written at runtime by [Code] (see GpuLogFlush), not
; declared in [Files] -- without this, the uninstaller wouldn't know about
; it and {app} could be left behind non-empty after uninstall.
Type: files; Name: "{app}\gpu_setup_log.txt"

[Code]
var
  GpuPage: TInputOptionWizardPage;
  GpuProgressPage: TOutputProgressWizardPage;

// index 0 = CPU only, 1/2/3 = CUDA 11/12/13
const
  GPU_CHOICE_NONE = 0;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if ProgressMax > 0 then
    GpuProgressPage.SetProgress(Progress, ProgressMax);
  Result := True;
end;

procedure InitializeWizard;
begin
  GpuPage := CreateInputOptionPage(wpSelectTasks,
    'GPU Acceleration', 'Choose whether to set up the GPU (CUDA) backend',
    'This downloads a small set of Python packages during install (needs internet access). ' +
    'It does NOT install the NVIDIA CUDA Toolkit itself -- that''s a separate, multi-GB ' +
    'download from NVIDIA and must already be installed and match the version you pick here ' +
    '(see developer.nvidia.com/cuda-downloads). If unsure, or if this machine has no NVIDIA ' +
    'GPU, choose "CPU only" -- you can always re-run this installer later to add GPU support.',
    True, False);
  GpuPage.Add('CPU only (no GPU) -- recommended if unsure');
  GpuPage.Add('NVIDIA GPU with CUDA 11.x installed');
  GpuPage.Add('NVIDIA GPU with CUDA 12.x installed');
  GpuPage.Add('NVIDIA GPU with CUDA 13.x installed');

  // /GPUCUDA=none|11|12|13 lets scripted/silent installs pick a value
  // that would otherwise only be reachable via this interactive page.
  case Lowercase(ExpandConstant('{param:GPUCUDA|}')) of
    '11': GpuPage.SelectedValueIndex := 1;
    '12': GpuPage.SelectedValueIndex := 2;
    '13': GpuPage.SelectedValueIndex := 3;
  else
    GpuPage.SelectedValueIndex := GPU_CHOICE_NONE;
  end;

  GpuProgressPage := CreateOutputProgressPage('Setting up GPU support',
    'Downloading and installing GPU packages -- this can take a minute...');
end;

function CudaPackageName(Choice: Integer): String;
begin
  case Choice of
    1: Result := 'cupy-cuda11x';
    2: Result := 'cupy-cuda12x';
    3: Result := 'cupy-cuda13x';
  else
    Result := '';
  end;
end;

var
  GpuLogLines: TArrayOfString;

procedure GpuLog(const S: String);
begin
  SetArrayLength(GpuLogLines, GetArrayLength(GpuLogLines) + 1);
  GpuLogLines[GetArrayLength(GpuLogLines) - 1] := GetDateTimeString('yyyy-mm-dd hh:nn:ss', #0, #0) + '  ' + S;
end;

procedure GpuLogOutput(const Prefix: String; const Output: TExecOutput);
var
  i: Integer;
begin
  for i := 0 to GetArrayLength(Output.StdOut) - 1 do
    GpuLog(Prefix + ' [stdout] ' + Output.StdOut[i]);
  for i := 0 to GetArrayLength(Output.StdErr) - 1 do
    GpuLog(Prefix + ' [stderr] ' + Output.StdErr[i]);
end;

procedure GpuLogFlush;
begin
  SaveStringsToFile(ExpandConstant('{app}\gpu_setup_log.txt'), GpuLogLines, False);
end;

// A bare MsgBox() blocks forever on a silent/unattended install (e.g.
// /VERYSILENT, or a scripted /GPUCUDA= run) -- /SUPPRESSMSGBOXES only
// suppresses Inno Setup's OWN built-in dialogs, not custom MsgBox calls
// from [Code], and there's no one there to click it. Confirmed this
// hangs a silent run indefinitely. GpuLogFlush already wrote the same
// information to gpu_setup_log.txt, so silent runs can skip the dialog
// entirely and still be fully diagnosable after the fact.
procedure GpuMsgBox(const Msg: String; Kind: TMsgBoxType; Buttons: Cardinal);
begin
  if not WizardSilent then
    MsgBox(Msg, Kind, Buttons);
end;

// Runs the bundled embeddable Python's pip to fetch cupy+cuda-pathfinder
// straight into {app}\_internal (PyInstaller's onedir import location),
// then downloads+extracts openpiv-python-gpu's openpiv_gpu package the
// same way. Failure here is reported but does NOT roll back the
// already-successful CPU install -- the app still works CPU-only.
//
// Writes {app}\gpu_setup_log.txt with every step's captured stdout/stderr
// -- confirmed on real hardware that this step can hit a TRANSIENT file
// lock (antivirus real-time scanning a just-downloaded file) rather than
// a real failure, so extraction retries up to 3 times with a delay
// before giving up; error MsgBoxes only ever showed an exit code before,
// with no way to tell a real failure from this kind of transient one.
procedure InstallGpuPackages(Choice: Integer);
var
  PyExe, PkgName, ZipPath, InternalDir: String;
  ResultCode, Attempt: Integer;
  DownloadOk, PipOk, ExtractOk: Boolean;
  Output: TExecOutput;
begin
  SetArrayLength(GpuLogLines, 0);
  PyExe := ExpandConstant('{tmp}\pyembed\python.exe');
  InternalDir := ExpandConstant('{app}\_internal');
  PkgName := CudaPackageName(Choice);
  GpuLog('Starting GPU setup for choice=' + IntToStr(Choice) + ' (' + PkgName + ')');

  if not FileExists(PyExe) then
  begin
    GpuLog('ERROR: bundled Python not found at ' + PyExe);
    GpuLogFlush;
    GpuMsgBox('GPU setup could not start -- the bundled Python tool is missing (' + PyExe + '). ' +
      'PIV Suite is installed and works CPU-only. See gpu_setup_log.txt in the install folder.',
      mbError, MB_OK);
    Exit;
  end;
  if PkgName = '' then Exit;

  GpuProgressPage.Show;
  try
    GpuProgressPage.SetText('Downloading ' + PkgName + ' (CUDA Python bindings)...', '');
    GpuLog('Running pip install --target "' + InternalDir + '" ' + PkgName + ' cuda-pathfinder');
    PipOk := ExecAndCaptureOutput(PyExe,
      '-m pip install --target "' + InternalDir +
      '" --no-warn-script-location --no-deps ' + PkgName + ' cuda-pathfinder',
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode, Output);
    GpuLogOutput('pip', Output);
    if not PipOk or (ResultCode <> 0) then
    begin
      GpuLog('ERROR: pip install failed, exit code ' + IntToStr(ResultCode));
      GpuLogFlush;
      GpuMsgBox('Downloading ' + PkgName + ' failed (exit code ' + IntToStr(ResultCode) + '). ' +
        'PIV Suite is installed and works CPU-only; you can re-run this installer to retry GPU setup. ' +
        'See gpu_setup_log.txt in the install folder for details.',
        mbError, MB_OK);
      Exit;
    end;
    GpuLog('pip install succeeded');

    GpuProgressPage.SetText('Downloading openpiv-python-gpu...', '');
    ZipPath := ExpandConstant('{tmp}\openpiv_gpu.zip');
    GpuLog('Downloading openpiv-python-gpu zip to ' + ZipPath);
    DownloadOk := DownloadTemporaryFile(
      'https://github.com/ali-sh-96/openpiv-python-gpu/archive/refs/heads/master.zip',
      'openpiv_gpu.zip', '', @OnDownloadProgress) > 0;
    if not DownloadOk then
    begin
      GpuLog('ERROR: openpiv-python-gpu download failed');
      GpuLogFlush;
      GpuMsgBox('Downloading openpiv-python-gpu failed -- check your internet connection. ' +
        'cupy installed OK, but the GPU backend needs both; re-run this installer to retry. ' +
        'See gpu_setup_log.txt in the install folder for details.',
        mbError, MB_OK);
      Exit;
    end;
    GpuLog('Download succeeded');

    GpuProgressPage.SetText('Extracting openpiv-python-gpu...', '');
    ExtractOk := False;
    for Attempt := 1 to 3 do
    begin
      GpuLog('Extraction attempt ' + IntToStr(Attempt));
      ExtractOk := ExecAndCaptureOutput(PyExe,
        'fetch_openpiv_gpu.py "' + ZipPath + '" "' + InternalDir + '"',
        ExpandConstant('{tmp}\pyembed'), SW_HIDE, ewWaitUntilTerminated, ResultCode, Output);
      GpuLogOutput('extract#' + IntToStr(Attempt), Output);
      ExtractOk := ExtractOk and (ResultCode = 0);
      if ExtractOk then
        Break;
      // A transient AV-scan lock on the just-downloaded zip is the known
      // cause here (confirmed manually on real hardware) -- back off and
      // retry rather than failing on the first attempt.
      GpuLog('Extraction attempt ' + IntToStr(Attempt) + ' failed (exit code ' + IntToStr(ResultCode) + '), retrying after delay');
      Sleep(2000);
    end;
    if not ExtractOk then
    begin
      GpuLog('ERROR: extraction failed after 3 attempts');
      GpuLogFlush;
      GpuMsgBox('Extracting openpiv-python-gpu failed after 3 attempts (exit code ' + IntToStr(ResultCode) + '). ' +
        'PIV Suite is installed and works CPU-only; you can re-run this installer to retry GPU setup. ' +
        'See gpu_setup_log.txt in the install folder for details.',
        mbError, MB_OK);
      Exit;
    end;
    GpuLog('Extraction succeeded');
    GpuLog('GPU setup completed successfully');
    GpuLogFlush;
  finally
    GpuProgressPage.Hide;
  end;

  GpuMsgBox('GPU support for ' + PkgName + ' installed successfully. Note this still requires a ' +
    'matching NVIDIA CUDA Toolkit already installed separately (developer.nvidia.com/cuda-downloads) ' +
    '-- the GPU option in PIV Suite will only work if that''s present and matches.',
    mbInformation, MB_OK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and (GpuPage.SelectedValueIndex <> GPU_CHOICE_NONE) then
    InstallGpuPackages(GpuPage.SelectedValueIndex);
end;
