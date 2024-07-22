import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
 
 
plt.figure(figsize=(25, 15))
mpl.rcParams['font.sans-serif'] = ['Times New Roman']  
mpl.rcParams['font.weight'] = 'bold'  
mpl.rcParams['font.size'] = 26  
plt.subplot(121)
fa1=[0.877,0.867,0.851,0.839,0.502]
fa2=[0.992,0.992,0.983,0.946,0.719]
fa3=[0.99,0.99,0.983,0.719,0.845]
index_fa1=[0.1,0.5,0.9,1.3,1.7]
index_fa2=[0.2,0.6,1.0,1.4,1.8]
index_fa3=[0.3,0.7,1.1,1.5,1.9]
plt.barh(index_fa1,fa1,height=0.1,label='Teemo',color='deepskyblue')
plt.barh(index_fa2,fa2,height=0.1,label='Yasuo',color='mediumturquoise')
plt.barh(index_fa3,fa3,height=0.1,label='Yone',color='g')
plt.legend(frameon=False,loc='upper right')
plt.ylim(0,2.3)
plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.yticks([0.2,0.6,1.0,1.4,1.8], ['None','SZA', 'SBTS', 'TCWV', 'BTS'], fontsize=26)
plt.ylabel('TA', fontsize=30,fontweight='bold')
plt.xlabel('ZX', fontsize=30,fontweight='bold')
plt.text(0, 2.35, '(a)')
ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
 
plt.subplot(122)
fb1=[0.799,0.826,0.815]# 4 7 10
fb2=[0.863,0.893,0.878]
fb3=[0.775,0.799,0.792]
fb4=[0.850,0.877,	0.865]
fb5=[0.77,0.760,0.806]
fb6=[0.847,0.873,0.877]
index_fb1=[1,2,3]
index_fb2=[1.2,2.2,3.2]
index_fb3=[1.1,2.1,3.1]
index_fb4=[1.3,2.3,3.3]
 
plt.bar(index_fb1,fb1,width=0.1,label='League Of Legend1',color='deepskyblue',zorder=1) 
plt.bar(index_fb2,fb2,width=0.1,label='League Of Legend2',color='deepskyblue',hatch='/',zorder=1)
plt.bar(index_fb3,fb3,width=0.1,label='League Of Legend3',color='mediumturquoise',zorder=1)
plt.bar(index_fb4,fb4,width=0.1,label='League Of Legend4',color='mediumturquoise',hatch='/',zorder=1)
plt.legend(frameon=False)
for i in range(3):
    plt.scatter(index_fb3[i], fb5[i], s=400, marker="*", color='black',zorder=2)
    plt.scatter(index_fb4[i], fb6[i], s=400, marker="*", color='black',zorder=2)
plt.ylim(0,1.3)
plt.xticks([1.15, 2.15, 3.15], ['April','July', 'October'], fontsize=26)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.scatter(2.2,1.01, s=400, marker="*", color='black',zorder=2)#120
plt.text(2.25, 1, 'MVP')
plt.text(0.8, 1.32, '(b)')
plt.xlabel('Test Month', fontsize=30,fontweight='bold')
plt.ylabel('Accuracy', fontsize=30,fontweight='bold')
ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
plt.savefig('./bar_example.pdf', bbox_inches='tight', dpi=1200)
plt.show()