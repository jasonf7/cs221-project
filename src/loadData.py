#!python
#!/usr/bin/env python
from scipy.io import loadmat
x = loadmat('/home/sarah/Downloads/DiffSeg-Data/mwu100307/data.mat')
# %%
print(x.keys())
# %% 
print(x['imgs'].shape)
print(x['segs'].shape)


import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,2)

# plt.show()
slide = 50
print(x['imgs'][:, :, slide, 1].min(), x['imgs'][:, :, slide, 1].max())
axarr[0,0].imshow(x['imgs'][:, :, slide, 0])
axarr[0,1].imshow(x['imgs'][:, :, slide, 1])
axarr[1,0].imshow(x['imgs'][:, :, slide, 2])
axarr[1,1].imshow(x['imgs'][:, :, slide, 3])
axarr[2,0].imshow(x['segs'][:, :, slide, 0],  cmap='plasma', vmin=0, vmax=77)
axarr[2,1].imshow(x['segs'][:, :, slide, 1],  cmap='plasma', vmin=0, vmax=2033)
# plt.colorbar()
plt.show()





# %%
