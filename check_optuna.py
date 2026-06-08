import optuna
import os

try:
    study = optuna.load_study(study_name="bayesian_research_a_study", storage="sqlite:///optuna_research_a.db")
    print(f"Total Trials: {len(study.trials)}")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Completed Trials: {len(completed)}")
    print(f"Pruned Trials: {len(pruned)}")
    
    if len(completed) > 0:
        print(f"Best Trial: {study.best_trial.number}")
        print(f"Best Edge Floor: {study.best_trial.value:.2f}")
        for k, v in study.best_trial.params.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.5f}")
            else:
                print(f"  {k}: {v}")
except Exception as e:
    print(f"Error loading study: {e}")
