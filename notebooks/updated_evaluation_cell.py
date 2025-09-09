# Updated Evaluation Cell - Replace the content in cell "6. Model Evaluation"

# Enhanced evaluation with better metrics and visualization
print("📊 Evaluating enhanced model performance...")

# Evaluate the trained model
eval_results = trainer_obj.evaluate_model(trainer, dataset)

print(f"\n📈 Enhanced Evaluation Results:")
print("=" * 50)
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Additional evaluation metrics
print(f"\n🔍 Training Summary:")
print(f"  Total training steps: {trainer.state.global_step}")
print(f"  Best model checkpoint: {trainer.state.best_model_checkpoint}")
print(f"  Training samples processed: {len(dataset['train']) * 5}")  # 5 epochs
print(f"  Final learning rate: {trainer.get_last_lr()[0] if trainer.get_last_lr() else 'N/A'}")

# Loss visualization (if training history is available)
if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
    print(f"\n📉 Training Progress:")
    train_losses = [log['train_loss'] for log in trainer.state.log_history if 'train_loss' in log]
    eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
    
    if train_losses:
        print(f"  Initial train loss: {train_losses[0]:.4f}")
        print(f"  Final train loss: {train_losses[-1]:.4f}")
        print(f"  Loss reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    
    if eval_losses:
        print(f"  Best eval loss: {min(eval_losses):.4f}")
        print(f"  Final eval loss: {eval_losses[-1]:.4f}")

print(f"\n✅ Enhanced evaluation completed!")
print(f"📝 Results saved to: results/evaluation_results.json")
