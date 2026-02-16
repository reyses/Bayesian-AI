
from training.orchestrator import _optimize_pattern_task, _optimize_template_task, simulate_trade_standalone
from training.orchestrator import FISSION_SUBSET_SIZE, INDIVIDUAL_OPTIMIZATION_ITERATIONS

def _process_template_job(args):
    """
    Multiprocessing Worker Function
    Executes the Fission/Optimization logic for a single template.
    Returns a result dict.
    """
    template, clustering_engine, iterations, generator, point_value = args

    # 1. Select Training Subset
    subset = template.patterns[:FISSION_SUBSET_SIZE]

    # 2. Run Individual Optimization (for Fission Check)
    member_optimals = []
    for pattern in subset:
        # Re-use existing standalone optimization task
        best_p, _ = _optimize_pattern_task((pattern, INDIVIDUAL_OPTIMIZATION_ITERATIONS, generator, point_value))
        member_optimals.append(best_p)

    # 3. Check for Behavioral Fission (Regret-Based)
    new_sub_templates = clustering_engine.refine_clusters(template.template_id, member_optimals, subset)

    if new_sub_templates:
        # FISSION DETECTED
        return {
            'status': 'SPLIT',
            'template_id': template.template_id,
            'new_templates': new_sub_templates
        }

    # 4. Consensus Optimization (No Fission)
    # Re-use existing standalone template optimization
    best_params, _ = _optimize_template_task((template, subset, iterations, generator, point_value))

    # 5. Validation
    val_pnl = 0.0
    validation_subset = template.patterns[FISSION_SUBSET_SIZE:]
    if validation_subset:
        # We need to simulate locally here since we are in a worker process
        for p in validation_subset:
             # We use the standalone simulation function
             outcome = simulate_trade_standalone(
                entry_price=p.price,
                data=p.window_data,
                state=p.state,
                params=best_params,
                point_value=point_value
            )
             if outcome:
                 val_pnl += outcome.pnl

    return {
        'status': 'DONE',
        'template_id': template.template_id,
        'template': template,
        'best_params': best_params,
        'val_pnl': val_pnl,
        'member_count': template.member_count
    }
