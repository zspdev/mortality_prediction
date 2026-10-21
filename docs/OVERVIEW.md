# Növeli-e a kórházi túlterheltség a COVID-halálozást? Közérthető áttekintés

## A kérdés
Ha a kórházak megtelnek, nagyobb eséllyel halnak-e meg a betegek? A járványt szimuláló ágensalapú modellünkben (PanSIM) a megfertőződő egészségügyi dolgozók (HCW) kiesnek a munkából, így csökken a kórházak tényleges kapacitása. Azt szeretnénk tudni, mennyivel emelkedjen a halálozási esély, ha a kapacitás elfogy, hogy a modell ezt tükrözni tudja.

## Az adatok
Heti kórházi adatok Belgiumból, Rio de Janeiróból, Spanyolországból, Magyarországról, Olaszországból és Törökországból: kórházban fekvők, intenzíven ápoltak, felvételek és halálozások. A török adatok kórházi része hiányos, ezért Törökország nem szerepel a modellekben. **Egészségügyi dolgozók fertőzöttségéről vagy távolmaradásáról nincs adat**, így a személyzeti hatást közvetlenül nem tudjuk mérni.

## A módszer
- **Terhelés:** minden kórház terhelése a saját legzsúfoltabb heteihez viszonyított betegszám (95. percentilis). A tényleges kapacitás ismeretlen, ez a közelítése.
- **Halálozás:** a heti halálozást a megelőző hetekben felvett betegekhez kötjük. A késleltetést a modell maga tanulja meg: a halálesetek nagy része kb. 1 héttel a felvétel után következik be, szinte mind 4–6 héten belül.
- **Magyarország és Olaszország:** itt nincs felvételi adat, ezért a halálozást a kórházban fekvők számához (betegheti) viszonyítottuk.
- **A szorzó:** azt számoljuk, mennyivel változik a halálozás kockázata a terhelés növekedésével, ha figyelembe vesszük, hogy a halálozás az idők során más okokból is változott (új variánsok, oltás, jobb kezelés).

## Eredmények
**Szorzó teljes terhelésnél** (az érték 1 = nincs hatás; az adott modellre számolt 90%-os sávval):

| Ország | Szorzó | Megjegyzés |
|---|---|---|
| Olaszország | kb. 1,7–1,9 | legerősebb hatás, a terhelés enyhén javítja az előrejelzést is |
| Belgium | kb. 1,5–1,8 | következetes növekedés, de csak 3 régió |
| Spanyolország | kb. 1,1–1,3 | gyenge hatás |
| Rio | kb. 1,0–1,5 | a bizonytalanság túl nagy |
| Magyarország | kb. 1,0–1,1 | nincs használható becslés (a sáv tartalmazza az 1-et) |

- **Az eredmény függ az időkezeléstől.** A nagy terhelésű hetek gyakran a korai, halálosabb hullámokra estek. Ha ezt nem vesszük figyelembe, a terhelés hatása nagynak látszik (kb. 1,6× teljes terhelésnél). Rugalmas időkorrekcióval ugyanez Spanyolországban kb. 1,0–1,2×-ra csökken.
- **Az országok eltérnek**, ezért nem átlagoljuk őket.
- **A terhelés alig javítja az előrejelzést** olyan időszakokra vagy régiókra, amelyeket a modell nem látott. Csak Olaszországban látszik mérsékelt javulás.
- **Az időbeli modellezés viszont sokat javított** a halálozások előrejelzésén (a késleltetés és az idővel változó halálozás figyelembevételével). Nem a modell, hanem a terhelés bevonása nem hozott megbízható többletet.

## Mit jelent ez az ABM számára?
- A szorzó **dinamikusan** használható: az ABM minden héten kiszámolja az aktuális terhelést, és kikeresi hozzá az országra jellemző szorzót. Alacsony terhelésnél a szorzó kb. 1, a hullám tetején 1,0 és kb. 1,9 között lehet.
- Az adatok nem döntik el, melyik görbe az igaz, ezért **forgatókönyveket** futtatunk: alacsony (kb. 1,0, Spanyolország), közepes (kb. 1,3) és magas (kb. 1,8, Olaszország és Belgium). A három szám a modelleredmények kerekítése, nem pontbecslés.
- **Magyarországra** a szorzót nem a magyar adat adja (nem informatív), hanem más országokból kölcsönzött görbe.
- A „HCW-kiesés → kevesebb kapacitás" kapcsolat az ABM feltevése (a kapacitás a kieső dolgozók arányában csökken). Az adat ezt nem támasztja alá, ezért ezt is érdemes variálni.

## Korlátok
- A kapacitás csak közelítés. Az életkor, az oltottság és a variáns nincs az adatokban, pedig ezek erősen befolyásolják a halálozást.
- A kórházi betegszám egyszerre a nevező és a terhelés mérőszáma, ezért az olasz hatás részben ebből adódhat (magas terhelésnél súlyosabb betegek vagy hosszabb ápolás).
- A legtöbb országban a halálozás nem csak kórházi.
- Kevés független régió (Belgium 3, Rio 1, Magyarország 1), ezért az országok közötti következtetések gyengék.

## Következő lépések
- Életkor szerinti, oltottsági és HCW-távolléti adat beszerzése, ha lehetséges. Ez jelentené a legnagyobb előrelépést.
- A szorzó beépítése a PanSIM-be (a `I6_h → D1` átmenetnél) és a három forgatókönyv érzékenységvizsgálata.

Technikai részletek: `docs/EXPERIMENTS.md`; terv: `PLAN.md`.
