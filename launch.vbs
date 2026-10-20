Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

projectPath = FSO.GetParentFolderName(WScript.ScriptFullName)

cmd = "cmd.exe /c cd /d """ & projectPath & """ && python -m streamlit run app.py"

WshShell.Run cmd, 0