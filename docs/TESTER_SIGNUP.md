# Pilot tester sign-up form: voice-donation section

Testers can agree to voice donation on the sign-up form instead of in the chat. The form must show **exactly** the
text below, which is the same text as donation version `2026-10-10` (`DONATION_VERSION`). Every box must be
**unticked** by default. Donation is optional: the tester can join the pilot without ticking anything in this
section.

The Hausa, Yoruba and Igbo texts are **DRAFT, pending native-speaker review**. Use them only after that review.
Two placeholders follow `KUDIGUARD_STORAGE_LOCATION`, using the same strings as the chat
(`data/languages/*.yaml`):
* **{where}:** "outside Nigeria during the pilot" (`where_outside`), or "in Nigeria" (`where_nigeria`).
* **{risk}:** while the server is abroad, "Nigerian data-protection law may not fully protect data held abroad."
  (`risk_outside`). Once it is in Nigeria, "Data passing through Meta and Cloudflare may be processed abroad, where
  Nigerian data-protection law may not fully apply." (`risk_nigeria`).

## Wording

**English**
> **Optional: help KudiGuard understand Nigerian voices.** If you agree, we keep the voice notes you send to
> KudiGuard from now on, to test and improve our speech recognition. A few trusted reviewers and transcribers, under
> written confidentiality, listen to them and write down what was said. We never use your voice to identify you,
> and we never sell or publish it. They are kept encrypted on our server {where} for up to 12 months, without your
> phone number. {risk} We try not to keep forwarded
> voice notes, and reviewers delete other people's voices. Send STOP DONATING to KudiGuard at any time to stop and
> delete them (STOP and DELETE also delete them). KudiGuard works the same whether you agree or not.
>
> ☐ I want to donate my voice notes as described above.
> ☐ I am 18 years or older.
> ☐ I will send only voice notes in my own voice, not recordings of other people.

**Hausa (DRAFT)**
> **Zaɓi ne: ku taimaka wa KudiGuard ta fahimci muryoyin 'yan Najeriya.** Idan kun amince, za mu adana saƙonnin
> murya da kuke aikowa KudiGuard daga yanzu, don gwada da inganta fahimtar magana. Wasu ƙalilan amintattun masu bita
> da masu rubutawa, ƙarƙashin yarjejeniyar sirri a rubuce, za su saurare su su rubuta abin da aka faɗa. Ba ma amfani
> da muryarku don gane ku, kuma ba ma sayar da ita ko wallafa ta. Ana adana su a ɓoye a sabarmu {where} har tsawon
> watanni 12, ba tare da lambar wayarku ba. {risk} Muna ƙoƙarin kada mu adana saƙonnin murya da aka tura, kuma masu bita suna share muryoyin wasu.
> Ku aika STOP DONATING zuwa KudiGuard a kowane lokaci don dakatarwa da share su (STOP da DELETE ma suna share su).
> KudiGuard tana aiki iri ɗaya ko kun amince ko a'a.
>
> ☐ Ina so in ba da gudummawar saƙonnin muryata kamar yadda aka bayyana a sama.
> ☐ Na kai shekara 18 ko fiye.
> ☐ Saƙonnin murya da muryata kawai zan aika, ba rikodin wasu mutane ba.

**Yoruba (DRAFT)**
> **Àṣàyàn ni: ẹ ran KudiGuard lọ́wọ́ láti gbọ́ ohùn àwọn ọmọ Nàìjíríà.** Tí ẹ bá gbà, a máa tọ́jú àtẹ̀jíṣẹ́ ohùn tí
> ẹ bá fi ránṣẹ́ sí KudiGuard láti ìsinsìnyí lọ, láti ṣàyẹ̀wò àti láti mú ìdámọ̀ ọ̀rọ̀ sísọ wa dára sí i. Àwọn
> olùyẹ̀wò àti akọ̀wé díẹ̀ tí a gbẹ́kẹ̀lé, lábẹ́ àdéhùn àṣírí tí a kọ sílẹ̀, ń gbọ́ wọn, wọ́n sì ń kọ ohun tí a sọ
> sílẹ̀. A kì í lo ohùn yín láti dá yín mọ̀, a kì í sì ta á tàbí tẹ̀ ẹ́ jáde. A ń tọ́jú wọn ní ìpamọ́ sórí sáfà wa
> {where} fún oṣù 12 jù lọ, láìsí nọ́mbà fóònù yín. {risk} A ń gbìyànjú láti má tọ́jú àtẹ̀jíṣẹ́ ohùn tí a dárí ránṣẹ́, àwọn olùyẹ̀wò sì ń pa ohùn àwọn ẹlòmíì rẹ́.
> Ẹ fi STOP DONATING ránṣẹ́ sí KudiGuard nígbàkígbà láti dá a dúró àti láti pa wọ́n rẹ́ (STOP àti DELETE náà ń pa
> wọ́n rẹ́). KudiGuard ń ṣiṣẹ́ bákan náà bóyá ẹ gbà tàbí ẹ kọ̀.
>
> ☐ Mo fẹ́ fi àtẹ̀jíṣẹ́ ohùn mi tọrẹ gẹ́gẹ́ bí a ti ṣàlàyé lókè.
> ☐ Mo ti pé ọmọ ọdún 18 tàbí jù bẹ́ẹ̀ lọ.
> ☐ Àtẹ̀jíṣẹ́ ohùn ara mi nìkan ni màá fi ránṣẹ́, kì í ṣe ohùn àwọn ẹlòmíì.

**Igbo (DRAFT)**
> **Ọ bụ nhọrọ: nyere KudiGuard aka ịghọta olu ndị Naịjirịa.** Ọ bụrụ na i kwere, anyị ga-echekwa ozi olu ị na-ezitere
> KudiGuard site ugbu a, iji nwalee ma melite nghọta okwu anyị. Ndị nyocha na ndị odee ole na ole a tụkwasịrị obi,
> n'okpuru nkwekọrịta nzuzo e dere ede, na-ege ha ntị ma dee ihe e kwuru. Anyị anaghị eji olu gị amata onye ị bụ,
> anyị anaghịkwa ere ya ma ọ bụ bipụta ya. A na-echekwa ha na nzuzo na sava anyị {where} ruo ọnwa 12, na-enweghị
> nọmba ekwentị gị. {risk} Anyị
> na-agbalị ka anyị ghara ichekwa ozi olu e zigharịrị, ndị nyocha na-ehichapụkwa olu ndị ọzọ. Zipu STOP DONATING
> na KudiGuard n'oge ọ bụla iji kwụsị ma hichapụ ha (STOP na DELETE na-ehichapụkwa ha). KudiGuard na-arụ ọrụ otu
> ihe ahụ ma i kwere ma ọ bụ jụ.
>
> ☐ Achọrọ m inye ozi olu m n'onyinye dịka a kọwara n'elu.
> ☐ Agbaala m afọ 18 ma ọ bụ karịa.
> ☐ Ọ bụ naanị ozi olu nke bụ olu m ka m ga-ezite, ọ bụghị ndekọ olu ndị ọzọ.

## Importing the answers
1. Export only the testers who ticked **all three** boxes. Use a CSV with these columns:
   * `number`: the WhatsApp number they will use;
   * `donate_confirmed`: yes or no;
   * `age_confirmed`: yes or no;
   * `own_voice_confirmed`: yes or no;
   * `consented_at`: when they submitted the form, in ISO format with a time zone, e.g. `2026-10-11T09:30:00+01:00`.

   **Format the `number` column as Text before you paste or type numbers.** A spreadsheet treats `08031234567` as a
   number. It silently drops the leading 0 (`8031234567`), and long numbers become `2.34803E+12`, which has lost
   digits. Accepted formats are `08031234567`, `0803 123 4567`, `+234 803 123 4567`, `2348031234567`,
   `00 234 803 123 4567`, and `8031234567` (the 0 already dropped). Anything else rejects the file.
2. Run `kudiguard voice preconsent --csv signups.csv --version 2026-10-10`.
   * The whole file is rejected, so nothing is half-imported, if any row has: a number that is not a Nigerian
     mobile; an unticked box; or a missing, zone-less or future time. Bad rows are listed by line number only.
   * The command prints how many testers were already known to KudiGuard and how many are new.
   * Some rows are skipped and reported. A form answer never overrides a later choice. Rows are skipped when the
     tester has since:
     * sent DELETE (a coded marker is kept for 30 days for exactly this);
     * sent STOP, STOP DONATING or "No";
     * been offered donation in the chat;
     * already said yes in the chat;
     * used KudiGuard after the form time.

     Rows are also skipped when the form was filled more than 29 days ago. A number listed twice counts once,
     with its most recent answer.
   * What happens to skipped testers:
     * Those never offered donation in the chat get the offer there later, as usual.
     * Those who were offered, declined, withdrew or stopped are not asked again. That is deliberate, to avoid
       nagging. They can opt in any time by sending **Donate voice**.
   * Import each export once, promptly. Answers older than 29 days are skipped, because DELETE markers last 30
     days.
3. Delete the CSV and the form export. They hold phone numbers. KudiGuard stores only the keyed hashes, with
   `source=form`.
4. The import does **not** sign anyone up for KudiGuard itself: each tester's first WhatsApp message still gets
   the welcome (retention, reviewers, transfer abroad), and they agree to it in the chat as everyone does.
   These testers are not offered donation again in the chat. Their voice notes are kept only while
   `KUDIGUARD_VOICE_DONATION=true`, and only those recorded in WhatsApp and not marked forwarded.
