# Storing your Brev credentials safely

Nothing in this repo reads, stores or prints a Brev token. `brev/run_on_brev.sh` only calls the `brev` CLI,
which uses whatever `brev login` saved. Keep it that way: **never put a token in a file in this repo, in
`env.yaml`, in a script, or in a chat message.**

## 1. Normal case: browser login (no token to handle)

```bash
brev login                 # opens the NVIDIA login page; the CLI stores the session itself
brev org ls                # confirm the org that holds your credit (marked with *)
```

The CLI keeps the session in `~/.brev/credentials.json`. Make sure only you can read it (it may be created
world-readable):

```bash
chmod 700 ~/.brev && chmod 600 ~/.brev/*.json
ls -l ~/.brev              # expect -rw------- for credentials.json
```

## 2. If you were given a token string (`brev login --token`)

Put it in the macOS Keychain, not in a file or your shell history:

```bash
# store (you are prompted for the value; it is not echoed or saved in history)
security add-generic-password -a "$USER" -s brev-token -w

# use it
brev login --token "$(security find-generic-password -a "$USER" -s brev-token -w)"

# remove it when the hackathon is over
security delete-generic-password -a "$USER" -s brev-token
```

The token is visible in the process list for the moment `brev login` runs, so run this on your own machine only.
On Linux use `secret-tool` (libsecret) or a password manager CLI (`pass`, 1Password `op read`) the same way.

## 3. Rules that matter more than the storage method

- **Never commit it.** `env.yaml`, `.env`, `*.key` and `*.pem` are already in `.gitignore`; keep tokens out of
  the repo entirely. Check with `git diff --cached | grep -i token` before committing.
- **Never paste it into a chat, issue or screenshot.** If you do, revoke it (log out of the Brev console /
  regenerate it) and log in again.
- **Do not copy `~/.brev` onto GPU instances.** The scripts do not; the instance never needs your credentials.
- Hugging Face tokens (not needed for the models used here, which are not gated) and any Red Hat registry
  password follow the same rule: Keychain, then `podman login registry.redhat.io --password-stdin` fed from it.
- **Limit the blast radius of a leak:** delete instances when you are done (`brev ls`, then
  `bash brev/run_on_brev.sh down`) and check the credit balance in the Brev console.
