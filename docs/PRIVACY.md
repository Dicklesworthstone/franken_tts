# FrankenTTS Privacy Policy

Effective date: August 10, 2026

FrankenTTS is designed to run text-to-speech and voice cloning locally on your device.
The app has no accounts, advertising, analytics, or tracking, and the FrankenTTS
developer does not collect personal data from the app.

## Data processed on your device

- Text you enter is used only for on-device speech synthesis.
- If you choose to clone a voice, the microphone recording is processed on-device into
  a small speaker embedding (a mathematical voice fingerprint). The recording is then
  discarded. The embedding is saved in the app's private storage until you delete that
  voice, and may be included in your device backups according to your iOS backup settings.
- If you import a FrankenTTS voice card from a picture, the selected picture is processed
  on-device and its embedded voice fingerprint is saved to your local voice library. The
  picture is not uploaded by FrankenTTS.
- Voice cards you choose to export contain the small voice fingerprint, not the original
  microphone recording. A person who receives a voice card can import and use that voice
  fingerprint. Share voice cards only for voices you have the right and consent to use.
- Synthesized audio and video remain on your device unless you explicitly use the system
  share sheet or save them to Photos.
- Downloaded model files are stored in the app's private storage and excluded from
  device backups.

FrankenTTS does not upload entered text, microphone recordings, speaker embeddings,
imported pictures, synthesized audio, or synthesized video.

## Network access

With your consent, the app downloads approximately 2 GB of model files from the
FrankenTTS project's GitHub Releases page. These requests do not contain your text,
recordings, speaker embedding, or synthesized audio. GitHub may receive routine network
request information, such as an IP address and user agent, under the
[GitHub Privacy Statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement).
The FrankenTTS developer does not receive this request information from the app.

## Permissions

- Microphone access is requested only when you start voice cloning. You can deny or
  revoke this permission in iOS Settings and continue using preset voices.
- Add-only Photos access is requested only when you choose to save a voice card or
  synthesized video to Photos.
- Importing a voice card uses Apple's system photo picker, which gives FrankenTTS access
  only to the picture you select rather than to your full photo library.

## Children

Because FrankenTTS does not collect personal data, it does not knowingly collect
personal data from children.

## Changes and contact

Material changes will be published in this document with a new effective date. For
privacy questions, open an issue in the
[FrankenTTS repository](https://github.com/Dicklesworthstone/franken_tts/issues).
