# ASR evaluation audio — sources and licences

| Folder / file | Source | Licence | Notes |
|---|---|---|---|
| `en_otp_tts.ogg` | Generated with Windows text-to-speech by the KudiGuard team | Own work | Synthetic; pipeline check only |
| `fleurs/ha`, `fleurs/yo`, `fleurs/ig` | [Google FLEURS](https://huggingface.co/datasets/google/fleurs) dev split (`ha_ng`, `yo_ng`, `ig_ng`) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | Re-encoded to Opus 16 kHz / 24 kbps (WhatsApp-style). Read speech; may overlap NCAIR1 training data, so treat scores as optimistic |

Citation: Conneau et al., "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech", 2022.
Real WhatsApp voice notes from testers (`source: whatsapp`) remain the primary evidence for the submission.
