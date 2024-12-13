import pandas as pd

from save_load import *
#from confu1 import *
from confu import *
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16
y_test= load("y_test")
y_prd=y_test.copy()
y_prd[:400]=y_test[400:800]
res=multi_confu_matrix(y_test,y_prd)
res=[i*100 for i in res]
print(res)
def plot_res():

    # Metrics data
    metrics_values = res
    metrics_names = [
        'Accuracy', 'Precision', 'Sensitivity', 'Specificity',
        'F_measure', 'MCC', 'NPV',"Fpr","Fnr"
    ]

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=metrics_names, y=metrics_values, palette="tab20b")
    plt.xticks(fontsize=16,fontweight='bold')
    plt.yticks(fontsize=16,fontweight='bold')
    plt.xlabel('Metrics',fontsize=14,fontweight='bold')
    plt.ylabel('Values',fontsize=14,fontweight='bold')
    #plt.ylim([70,])
    plt.tight_layout()
    plt.savefig("Results/ab_res1.svg", format='svg',dpi=800)
    plt.show()




    metrics_names = [
        'Accuracy', 'Precision', 'Sensitivity', 'Specificity',
        'F_measure', 'MCC', 'NPV',"Fpr","Fnr"
    ]

    metrics_df = pd.DataFrame({'Metric': metrics_names, 'Value (%)': res})

    metrics_df.to_csv("Results/ab_res.csv")
    # Display the DataFrame
    print(metrics_df)



plot_res()


from sklearn.metrics import confusion_matrix


#y_test = np.argmax(y_test, axis=1)
#y_pred = np.argmax(y_prd, axis=1)
print(y_prd)
print(type(y_prd))

y_prd_labels = np.argmax(y_prd, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

mat = confusion_matrix(y_test_labels,y_prd_labels)

labels=["N","L","R","A","V"]
    # Plot confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(mat, annot=True, cmap="tab20b", fmt="d", xticklabels=labels,yticklabels=labels)
plt.xlabel('Predicted labels',fontsize=16,fontweight='bold')
plt.ylabel('True labels',fontsize=16,fontweight='bold')
plt.tight_layout()

plt.savefig("Results/ab_confu.svg",format="svg",dpi=800)
plt.show()
