# Contract-roll spread — ATLAS

Levels do not transfer as raw numbers across a roll; they transfer once shifted by the calendar spread. This measures the shift.

## Control — 254 non-roll session boundaries

- mean `+0.0118%` · median `+0.0053%` · sd `0.0866%`
- 95% of boundaries within `[-0.207%, +0.189%]`

## Roll boundaries

| boundary | contracts | jump (pt) | jump (%) | z vs control |
|---|---|---|---|---|
| 2024_03_08 → 2024_03_11 | MNQH4 → MNQM4 | +245.00 | +1.357% | +15.5 |
| 2024_06_14 → 2024_06_17 | MNQM4 → MNQU4 | +258.50 | +1.312% | +15.0 |
| 2024_09_13 → 2024_09_16 | MNQU4 → MNQZ4 | +230.25 | +1.181% | +13.5 |
| 2024_12_13 → 2024_12_16 | MNQZ4 → MNQH5 | +292.50 | +1.344% | +15.4 |

All 4 rolls sit z ≥ +13 from the non-roll distribution — the offset is unambiguous, not a judgement call.

## Translation onto 2024_09_16 (MNQZ4)

- prior session 2024_09_13 (MNQU4) settle `19504.00`
- observed jump `+230.25pt` (`+1.181%`), z=`+13.5`
- **spread ≈ `+227.9pt`**, 95% `[+194.8, +261.1]` — the band is the overnight move, which cannot be removed

| prior-contract level | → this contract | 95% band |
|---|---|---|
| 19577.00 | **19804.95** | [19771.83, 19838.07] |
| 19396.00 | **19623.95** | [19590.83, 19657.07] |
| 19504.00 | **19731.95** | [19698.83, 19765.07] |
| 19683.00 | **19910.95** | [19877.83, 19944.07] |

