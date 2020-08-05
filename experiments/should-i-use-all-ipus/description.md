Should I use all the IPUs in a multi-IPU system even when my problem fits on 1?
==========================

When we are using an IPU system in its multi-IPU configuration, the programming model abstracts away 
the fact that there are several IPUs in use by creating one large ‘virtual IPU’. However, this abstraction
 does not take into account the higher cost of communication between tiles on different IPUs vs on the same IPU. 

This is only a problem when exchanges must take place between iterations of the stencil and would not
 apply to an embarrassingly parallel problem (where it is best to always use all the available processors).
  For a problem with frequent exchanges, when a given input fits on a single IPU, we want to know whether 
  the increased parallelism from using more than 1 IPU outweighs the increased communication cost.
  
We run a stencil against a 2200x1122 grid where each cell uses 32 bytes. This problem can fit on 1
 IPU, but we also run it on 2,4,8 and 16 IPUs and show the run time (average of 5 runs). We can clearly 
 see that, despite the increased cost of communication when exchanging outside of 1 IPU, the increased parallelism
  results in a faster execution time. Therefore, we should always try to use the maximum IPUs available, even when
   the problem fits on fewer.

But note the vastly increased compile times for more IPUs!

Plotting
-----
```python
df = pd.read_csv('experiments/should-i-use-all-ipus/results.csv')
df = df.groupby('NumIpu').mean().reset_index()
fig, ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(df.NumIpu, df.RunTime_s, 'x-', label='Run time (s)')
ax.set_xlabel('Number of IPUs',**{ 'fontsize': 16})
ax.set_ylabel('Run time(s)',**{ 'fontsize': 16})
ax.set_xscale('log')
ax.set_xticks([1,2,4,8,16])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_title('Runtime as  number of IPUs is increased',**{ 'fontsize': 24});
ax.grid()
plt.tight_layout()
```

