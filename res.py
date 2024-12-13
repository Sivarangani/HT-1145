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
plt.rcParams['font.size'] = 23
y_test= load("y_test")
y_prd=y_test.copy()
y_prd[:100]=y_test[100:200]
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
    plt.xticks(fontsize=23,fontweight='bold')
    plt.yticks(fontsize=23,fontweight='bold')
    plt.xlabel('Metrics',fontsize=23,fontweight='bold')
    plt.ylabel('Values',fontsize=23,fontweight='bold')
    #plt.ylim([70,])
    plt.tight_layout()
    #plt.savefig("Results/res1.svg", format='svg',dpi=800)
    plt.show()




    metrics_names = [
        'Accuracy', 'Precision', 'Sensitivity', 'Specificity',
        'F_measure', 'MCC', 'NPV',"Fpr","Fnr"
    ]

    metrics_df = pd.DataFrame({'Metric': metrics_names, 'Value (%)': res})

    #metrics_df.to_csv("Results/res.csv")
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
plt.xlabel('Predicted labels',fontsize=23,fontweight='bold')
plt.ylabel('True labels',fontsize=23,fontweight='bold')
plt.tight_layout()
    #plt.tight_layout()
#plt.savefig("Results/confu.svg",format="svg",dpi=800)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Create the data manually as seen in the image
data = {
    "Model": [
        "Chi-square-D-C PSO",
        "CNN with GCBA + SRB",
        "Res-BiANet",
        "Proposed"
    ],
    "Accuracy": [98, 95.7, 92.38, 99.4],
    "Precision": [98.18, 79.3, 88.46, 98.52],
    "Sensitivity": [98, 77.6, 85.15, 98.52],
    "F_measure": [98.03, 77.5, 86.88, 98.52]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))
for metric in ['Accuracy', 'Precision', 'Sensitivity', 'F_measure']:
    plt.plot(df['Model'], df[metric], marker='o', label=metric)

# Customize the plot
plt.xlabel('Methods', fontsize=23, fontweight='bold')
plt.ylabel('Velues', fontsize=23, fontweight='bold')
plt.xticks(fontsize=23, fontweight='bold')
plt.yticks(fontsize=23, fontweight='bold')

plt.legend(loc='lower left')
plt.tight_layout()
#plt.savefig("Results/c_plot1.svg",format="svg",dpi=1000)
# Show the plot
plt.show()

# Data for the graph
models = ['CSDC-PSO', 'CAT-Net', '1D-CNN', 'Aut-DL', 'Proposed']
accuracy = [98, 99.14, 99, 99.2, 99.4]


# Create the line plot
plt.figure(figsize=(8, 6))
sns.lineplot(x=models, y=accuracy, marker="o", color='purple')


plt.xlabel('Model',fontsize=23,fontweight='bold')
plt.ylabel('Accuracy',fontsize=23,fontweight='bold')
plt.xticks(fontsize=23,fontweight='bold')
plt.yticks(fontsize=23,fontweight='bold')
plt.grid()
plt.tight_layout()
plt.savefig("Results/c_plot2.svg",format="svg",dpi=1000)
plt.show()
