# Publish the full source tree

1. Download the sanitized source ZIP attached in the project conversation.
2. Extract it to a local folder.
3. Open PowerShell in the extracted project root.
4. Ensure Git for Windows is installed and authenticated to GitHub.
5. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\publish.ps1
```

The script clones this repository, copies the sanitized source tree, creates a source commit, and pushes it to `main`. It excludes local environments, local configuration, generated datasets, and run outputs.

The source ZIP is the complete code package; this GitHub landing commit contains the README and technical overview until the source-publish script is run.
