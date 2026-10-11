# Demo script (3 minutes)

1. **Problem (20 s)**: OTP and "account blocked" scams cost Nigerians billions; many targets are more comfortable in
   Hausa, Yoruba or Igbo, and by voice.
2. **First contact (20 s)**: send "Sannu" → welcome + language menu in Hausa → pick Hausa.
3. **Voice check (45 s)**: send a Hausa voice note about a "bank staff" call asking for the OTP →
   🎧 Listening… → 🔴 *Wannan yana kama da DAMFARA* + "I heard…" + steps + verified report channels.
4. **Buttons (30 s)**: *Me zan yi?* (steps) → *Gargaɗi iyalina* (forwardable family warning) → 👍.
5. **Another language (20 s)**: switch to Yorùbá; forward a fake "you won ₦2m, pay a fee" SMS → 🔴.
6. **Honesty (15 s)**: a genuine money question → helpful answer, no verdict; the fixed texts never say "it's safe".
7. **Scale (20 s)**: same engine via `POST /v1/check` (show `/docs`) for banks and fintechs; web page fallback. The
   partner API is off in the pilot: for the demo, run a local instance with `KUDIGUARD_PARTNER_API=true` and send an
   `X-KudiGuard-End-User` id (each end user counts toward the licence cap).
8. **Evidence (10 s)**: `kudiguard stats` → real users, languages, helpful %, latency.

Pre-flight: watchdog running, `/readyz` ok, test phone on Wi-Fi, a Hausa voice note recorded in advance as backup.
