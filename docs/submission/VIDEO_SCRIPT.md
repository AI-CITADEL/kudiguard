# Video script (3–5 minutes, web voice page)

Record the screen (Windows: Win+Alt+R, or OBS) with the microphone on. One take per scene is fine; join them after.
Show the real public link in the address bar. Speak slowly; keep each voice note 5–20 seconds.

| # | Time | Show | Say (voice-over, English) |
|---|---|---|---|
| 1 | 0:00–0:25 | Title card: "KudiGuard — check a suspicious call or message in your language" | "Every day Nigerians get calls saying their BVN is blocked, or asking for an OTP. Many are more comfortable in Hausa, Yoruba or Igbo, and by voice. KudiGuard checks it for them, on N-ATLAS." |
| 2 | 0:25–1:10 | Web page → choose **Hausa** → record: *"Wani ya kira ni ya ce an toshe BVN ɗina, ya ce in tura masa lambar OTP"* | "A real user speaks Hausa. NCAIR's Hausa speech model transcribes it, N-ATLAS reads it, and safety rules check it." Pause on: "I heard…", the 🔴 verdict, the steps, the report channels. |
| 3 | 1:10–1:50 | Switch to **Yoruba** → record: *"Ẹnìkan pè mí, ó ní kí n fi BVN mi ránṣẹ́ sí òun"* | "Same in Yoruba. The speech model sometimes spells bank words by sound — 'bifi ẹni' for BVN — so KudiGuard corrects those before N-ATLAS reads them." Show the 🔴 reply. |
| 4 | 1:50–2:20 | **Typed** genuine question in English or Pidgin: *"How do I check my BVN?"* | "It also answers everyday money questions from a curated knowledge base, and it never tells anyone a message is 'safe'." |
| 5 | 2:20–2:50 | A real tester (ideally on camera or their screen), in their language | "We tested live today with Hausa and Yoruba speakers, including testers outside Nigeria. Their messages exposed real problems, and we fixed them the same day." |
| 6 | 2:50–3:30 | Terminal: `kudiguard stats`; then `docs/submission/02_real_world_validation.pdf` (live table, fixes) | "Every check is logged privately — no phone numbers kept. Here are today's numbers, the failures we found, and the fixes. On our frozen 366-message test set it catches 91% of scams; over-flagging genuine bank notices is the main weakness, and we say so." |
| 7 | 3:30–4:00 | Architecture slide or `03_technical_documentation.pdf` diagram; `/docs` API page | "Voice in, N-ATLAS and NCAIR speech on our own GPU, safety rules around the model, and the same engine behind an API for banks and fintechs. Next: spoken replies, which testers asked for, and native-speaker review." |
| 8 | 4:00–4:15 | End card: link, team name | "KudiGuard. Built on N-ATLAS for NAIC 2026." |

**Before recording:** `https://<link>/readyz` shows `"status":"ok"`; send one throwaway check first (warms the model);
close other heavy apps; have a backup voice note recorded in case of noise.
**Do not show:** `.env`, tokens, phone numbers, the database file, tester names (unless they agreed).
Upload as **unlisted** YouTube (or Drive with "anyone with the link") and test the link in a private window.
