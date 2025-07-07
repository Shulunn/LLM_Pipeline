# 我要写一个两个list的分类评估的脚本，
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        'accuracy': accuracy,
        'classification_report': report
    }

#请你帮我写个example
def example_evaluation():
    # Example true labels and predicted labels
    y_true = ['cat', 'dog', 'cat', 'cat', 'dog', 'dog']
    y_pred = ['cat', 'dog', 'dog', 'cat', 'dog', 'cat']

    # Evaluate the classification results
    results = evaluate_classification(y_true, y_pred)
    #然后绘制混淆矩阵图
    # Note: You can use matplotlib or seaborn to visualize the confusion matrix if needed.

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['cat', 'dog'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['cat', 'dog'], yticklabels=['cat', 'dog'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    # Print the results
    print(f"Accuracy: {results['accuracy']}")
    print("Classification Report:")
    print(results['classification_report'])

# Example usage of the evaluation function
if __name__ == '__main__':
    # Run the example evaluation
    example_evaluation()

    # 这里可以替换成实际的y_true和y_pred列表进行评估
    # y_true = [...]
    # y_pred = [...]
    # results = evaluate_classification(y_true, y_pred)
    # print(results)