import numpy as np
import matplotlib.pyplot as plt
f1 = open('logout4.4.txt', 'r')

lns = f1.readlines()

l1 = []
l2 = []
for ln in lns:
    d = ln.split('- loss: ')
    if len(d) >= 2:
        d = d[1][:8]
        l1.append(d)
    d = ln.split(' from ')
    if len(d) >= 2:
        d = d[1][:8]
        l2.append(d)

plt.figure(num=3)
plt.plot([i for i in range(len(l1))], l1)

plt.plot([(i+1)*967 for i in range(len(l2))], l2, color='red', linewidth=1.0, linestyle='--')

plt.show()
# plt.savefig('demo.png')
print(len(l1)/967, len(l2))