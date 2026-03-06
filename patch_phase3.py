import re
import os

with open('training/orchestrator.py', 'r') as f:
    content = f.read()

# 1. Add direction correction accumulator before the Final Report
correction_str = """
        # ═══════════════════════════════════════════════════════════════════
        # ORACLE DIRECTION LEARNING (supervised correction)
        # ═══════════════════════════════════════════════════════════════════

        if oracle_trade_records and not oos_mode:
            print("\\n  Learning direction corrections from oracle...")

            _dir_corrections = defaultdict(lambda: {
                'long_correct': 0, 'long_wrong': 0,
                'short_correct': 0, 'short_wrong': 0,
                'long_pnl': 0.0, 'short_pnl': 0.0,
                'signed_mfe_samples': [],
            })

            for rec in oracle_trade_records:
                tid = rec.get('template_id')
                if tid is None or tid == -1:
                    continue

                direction = rec.get('direction', '')
                oracle_label = rec.get('oracle_label', 0)
                actual_pnl = rec.get('actual_pnl', 0.0)
                oracle_mfe = rec.get('oracle_mfe', 0.0)
                oracle_mae = rec.get('oracle_mae', 0.0)

                acc = _dir_corrections[tid]
                oracle_says_long = oracle_label > 0
                oracle_says_short = oracle_label < 0

                if direction == 'LONG':
                    acc['long_pnl'] += actual_pnl
                    if oracle_says_long:
                        acc['long_correct'] += 1
                    elif oracle_says_short:
                        acc['long_wrong'] += 1

                if direction == 'SHORT':
                    acc['short_pnl'] += actual_pnl
                    if oracle_says_short:
                        acc['short_correct'] += 1
                    elif oracle_says_long:
                        acc['short_wrong'] += 1

                if oracle_label != 0:
                    signed_mfe = oracle_mfe if oracle_label > 0 else -oracle_mae
                    acc['signed_mfe_samples'].append({
                        'signed_mfe': signed_mfe,
                        'entry_depth': rec.get('entry_depth', 6),
                        'dmi_diff': rec.get('dmi_diff', 0.0),
                        'oracle_label': oracle_label,
                    })

            _updated_count = 0
            _regression_count = 0

            for tid, acc in _dir_corrections.items():
                if tid not in self.pattern_library:
                    continue
                lib = self.pattern_library[tid]

                # ── Corrected direction bias (70% forward pass, 30% original) ──
                long_total = acc['long_correct'] + acc['long_wrong']
                short_total = acc['short_correct'] + acc['short_wrong']
                total_dir_trades = long_total + short_total

                if total_dir_trades >= 3:
                    fp_long_correct = acc['long_correct']
                    fp_short_correct = acc['short_correct']
                    fp_total_correct = fp_long_correct + fp_short_correct

                    fp_long_bias = fp_long_correct / fp_total_correct if fp_total_correct > 0 else 0.5
                    fp_short_bias = fp_short_correct / fp_total_correct if fp_total_correct > 0 else 0.5

                    orig_long = lib.get('long_bias', 0.5)
                    orig_short = lib.get('short_bias', 0.5)

                    new_long = 0.7 * fp_long_bias + 0.3 * orig_long
                    new_short = 0.7 * fp_short_bias + 0.3 * orig_short
                    total = new_long + new_short
                    if total > 0:
                        new_long /= total
                        new_short /= total

                    lib['long_bias'] = round(new_long, 4)
                    lib['short_bias'] = round(new_short, 4)
                    lib['direction_source'] = 'oracle_corrected'
                    _updated_count += 1

                # ── PnL-weighted direction signal ──
                if long_total >= 2 and short_total >= 2:
                    lib['long_avg_pnl'] = round(acc['long_pnl'] / long_total, 2)
                    lib['short_avg_pnl'] = round(acc['short_pnl'] / short_total, 2)

                # ── Signed MFE regression ──
                samples = acc['signed_mfe_samples']
                if len(samples) >= 15:
                    try:
                        from sklearn.linear_model import LinearRegression
                        X = np.array([[s['entry_depth'], s['dmi_diff']] for s in samples])
                        y = np.array([s['signed_mfe'] for s in samples])
                        reg = LinearRegression().fit(X, y)
                        lib['signed_mfe_coeff'] = reg.coef_.tolist()
                        lib['signed_mfe_intercept'] = float(reg.intercept_)
                        _regression_count += 1
                    except Exception:
                        pass

            print(f"  Direction corrections: {_updated_count} templates updated")
            print(f"  Signed MFE regression: {_regression_count} templates fitted")

            import pickle as _pkl_dir
            _lib_path = os.path.join(self.checkpoint_dir, 'pattern_library.pkl')
            with open(_lib_path, 'wb') as _f:
                _pkl_dir.dump(self.pattern_library, _f)
            print(f"  Updated pattern_library.pkl saved")

        # Final Report
"""
content = re.sub(r'# Final Report', correction_str, content)

# 2. Priority 0.5 and 1.5 in training/orchestrator.py
priority_str = """
                            # Priority 0.5: Signed MFE regression (learned from IS forward pass)
                            if side is None:
                                _smfe_coeff = lib_entry.get('signed_mfe_coeff')
                                if _smfe_coeff is not None:
                                    _entry_depth = getattr(best_candidate, 'depth', 6)
                                    _live_dmi = (getattr(best_candidate.state, 'dmi_plus', 0.0)
                                               - getattr(best_candidate.state, 'dmi_minus', 0.0))
                                    _smfe_features = np.array([[_entry_depth, _live_dmi]])
                                    _pred_smfe = float(
                                        np.dot(_smfe_features, np.array(_smfe_coeff))
                                        + lib_entry.get('signed_mfe_intercept', 0.0)
                                    )
                                    if abs(_pred_smfe) > 0.5:
                                        side = 'long' if _pred_smfe > 0 else 'short'

                            # Priority 1.0: per-cluster logistic regression P(LONG)
"""
content = content.replace("# Priority 1: per-cluster logistic regression P(LONG)", priority_str)

priority_15_str = """
                            # Priority 1.5: Brain direction-specific win rate
                            if side is None:
                                _dir_long_prob = self.brain.get_dir_probability(best_tid, 'LONG') if hasattr(self.brain, 'get_dir_probability') else None
                                _dir_short_prob = self.brain.get_dir_probability(best_tid, 'SHORT') if hasattr(self.brain, 'get_dir_probability') else None
                                if _dir_long_prob is not None and _dir_short_prob is not None:
                                    if _dir_long_prob > _dir_short_prob + 0.10:
                                        side = 'long'
                                    elif _dir_short_prob > _dir_long_prob + 0.10:
                                        side = 'short'

                            # Priority 2: template aggregate bias
"""
content = content.replace("# Priority 2: template aggregate bias", priority_15_str)

# 3. Add save pattern_forward_brain.pkl
save_brain_str = """
        if not oos_mode:
            _brain_path = os.path.join(self.checkpoint_dir, 'pattern_forward_brain.pkl')
            self.brain.save(_brain_path)
            print(f"  Forward pass brain saved: {_brain_path}")
            print(f"    States: {len(self.brain.table)}")
            if hasattr(self.brain, 'dir_table'):
                print(f"    Direction pairs: {len(self.brain.dir_table)}")

        print("\\n  [OK] Forward pass complete -- all files saved.", flush=True)
"""
content = content.replace("print(\"\\n  [OK] Forward pass complete -- all files saved.\", flush=True)", save_brain_str)

# 4. Add Direction Learning Report to Final Report Section
report_str = """
        if _dir_corrections:
            report_lines.append("")
            report_lines.append("=" * 80)
            report_lines.append("DIRECTION LEARNING (oracle corrections absorbed)")
            report_lines.append("=" * 80)

            _total_corrected = sum(
                1 for acc in _dir_corrections.values()
                if (acc['long_correct'] + acc['long_wrong'] +
                    acc['short_correct'] + acc['short_wrong']) >= 3
            )
            _total_smfe = sum(
                1 for acc in _dir_corrections.values()
                if len(acc['signed_mfe_samples']) >= 15
            )

            report_lines.append(f"  Templates with direction corrections: {_total_corrected}")
            report_lines.append(f"  Templates with signed MFE regression: {_total_smfe}")

            _corrections_list = []
            for tid, acc in _dir_corrections.items():
                if tid not in self.pattern_library:
                    continue
                lib = self.pattern_library[tid]
                orig_long = lib.get('long_bias', 0.5)
                long_total = acc['long_correct'] + acc['long_wrong']
                short_total = acc['short_correct'] + acc['short_wrong']
                if long_total + short_total < 3:
                    continue
                _corrections_list.append({
                    'tid': tid,
                    'orig_long_bias': orig_long,
                    'new_long_bias': lib.get('long_bias', 0.5),
                    'long_correct': acc['long_correct'],
                    'long_wrong': acc['long_wrong'],
                    'short_correct': acc['short_correct'],
                    'short_wrong': acc['short_wrong'],
                    'long_pnl': acc['long_pnl'],
                    'short_pnl': acc['short_pnl'],
                    'shift': abs(lib.get('long_bias', 0.5) - orig_long),
                })

            _corrections_list.sort(key=lambda x: -x['shift'])

            if _corrections_list:
                report_lines.append("")
                report_lines.append(f"  TOP 15 DIRECTION CORRECTIONS (biggest bias shift):")
                report_lines.append(f"  {'TID':>8} {'Orig':>6} {'New':>6} {'Shift':>6} "
                                   f"{'L_ok':>5} {'L_bad':>6} {'S_ok':>5} {'S_bad':>6} "
                                   f"{'L_PnL':>10} {'S_PnL':>10}")
                for r in _corrections_list[:15]:
                    report_lines.append(
                        f"  {str(r['tid']):>8} {r['orig_long_bias']:>6.2f} "
                        f"{r['new_long_bias']:>6.2f} {r['shift']:>+5.2f} "
                        f"{r['long_correct']:>5} {r['long_wrong']:>6} "
                        f"{r['short_correct']:>5} {r['short_wrong']:>6} "
                        f"${r['long_pnl']:>9,.0f} ${r['short_pnl']:>9,.0f}")

            _all_long_ok = sum(a['long_correct'] for a in _dir_corrections.values())
            _all_long_bad = sum(a['long_wrong'] for a in _dir_corrections.values())
            _all_short_ok = sum(a['short_correct'] for a in _dir_corrections.values())
            _all_short_bad = sum(a['short_wrong'] for a in _dir_corrections.values())
            _all_total = _all_long_ok + _all_long_bad + _all_short_ok + _all_short_bad
            _all_correct = _all_long_ok + _all_short_ok

            if _all_total > 0:
                report_lines.append("")
                report_lines.append(f"  DIRECTION ACCURACY (this run):")
                report_lines.append(f"    Correct: {_all_correct}/{_all_total} "
                                   f"({_all_correct/_all_total*100:.1f}%)")
                report_lines.append(f"    LONG  correct: {_all_long_ok}  wrong: {_all_long_bad}")
                report_lines.append(f"    SHORT correct: {_all_short_ok}  wrong: {_all_short_bad}")

        # Store for bottom-line summary at program exit
"""
content = content.replace("# Store for bottom-line summary at program exit", report_str)

with open('training/orchestrator.py', 'w') as f:
    f.write(content)

# Update live/live_engine.py
with open('live/live_engine.py', 'r') as f:
    live_content = f.read()

live_brain_str = """
        live_brain_path = os.path.join(cpdir, 'live_brain.pkl')
        forward_brain_path = os.path.join(cpdir, 'pattern_forward_brain.pkl')
        training_brains = sorted(glob.glob(os.path.join(cpdir, 'pattern_*_brain.pkl')))

        if os.path.exists(live_brain_path):
            self._brain.load(live_brain_path)
            logger.info(f"  Brain: live_brain.pkl ({len(self._brain.table)} states)")
        elif os.path.exists(forward_brain_path):
            self._brain.load(forward_brain_path)
            logger.info(f"  Brain: pattern_forward_brain.pkl ({len(self._brain.table)} states) — IS-learned directions")
        elif training_brains:
            self._brain.load(training_brains[-1])
            logger.info(f"  Brain: {os.path.basename(training_brains[-1])} (training base)")
        else:
            logger.warning("  No brain checkpoint found — starting fresh "
                          "(will learn from live trades)")
"""
live_content = re.sub(r'live_brain_path = os\.path\.join\(cpdir, \'live_brain\.pkl\'\)\n[\s\S]*?\(will learn from live trades\)"\)', live_brain_str, live_content)

# add Priority 0.5 in live_engine.py
live_priority_str = """
        # Priority 0.5: Brain direction-specific win rate
        if hasattr(self._brain, 'get_dir_probability'):
            _dir_long = self._brain.get_dir_probability(base_tid, 'LONG')
            _dir_short = self._brain.get_dir_probability(base_tid, 'SHORT')
            if _dir_long is not None and _dir_short is not None:
                if _dir_long > _dir_short + 0.10:
                    return 'long', _dir_long, 'brain_dir'
                elif _dir_short > _dir_long + 0.10:
                    return 'short', 1.0 - _dir_short, 'brain_dir'

        # Priority 1: live momentum (velocity + acceleration from physics engine)
"""
live_content = live_content.replace("# Priority 1: live momentum (velocity + acceleration from physics engine)", live_priority_str)

with open('live/live_engine.py', 'w') as f:
    f.write(live_content)
