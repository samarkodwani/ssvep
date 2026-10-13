import numpy as np
import matplotlib.pyplot as plt
from pipeline import load_subject
from cca import classify

SUBJECTS = [1, 5, 15, 18, 29]
START = 285                                  # sample at t = 0.64 s
WINS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]      # window lengths in seconds

acc = np.zeros((len(SUBJECTS), len(WINS)))

for si, s in enumerate(SUBJECTS):
    x, labels, freqs = load_subject(s)
    for j, w in enumerate(WINS):
        n = int(w * 250)
        pred = [classify(x[i, :, START:START + n], freqs) for i in range(240)]
        acc[si, j] = np.mean(np.array(pred) == labels)
    print(f'S{s}', acc[si].round(2))

np.save('accuracy.npy', acc)

for si, s in enumerate(SUBJECTS):
    plt.plot(WINS, acc[si] * 100, alpha=0.4, label=f'S{s}')
plt.plot(WINS, acc.mean(0) * 100, 'k', lw=2, label='mean')
plt.axhline(2.5, color='grey', ls=':', label='chance')
plt.xlabel('window length (s)')
plt.ylabel('accuracy (%)')
plt.legend()
plt.savefig('accuracy_vs_window.png', dpi=150)
plt.show()