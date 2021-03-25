length = len(self.snippets[0].probes[0].FID.diffPhase_filtered) + len(self.snippets[1].probes[0].FID.diffPhase_filtered) + len(
    self.snippets[2].probes[0].FID.diffPhase_filtered)
mat = np.matrix((np.zeros(length), np.zeros(length), np.zeros(length)))

for i in np.arange(0, len(self.snippets[0].probes) - 1):
    a = self.snippets[0].probes[i].FID.diffPhase_filtered
    b = self.snippets[1].probes[i].FID.diffPhase_filtered
    c = self.snippets[2].probes[i].FID.diffPhase_filtered
    conc = np.concatenate((a, b, c))
    mat[i, :] = conc
diffSnippets = mat.transpose()

x1 = self.snippets[0].probes[0].xPos[1, 0]
x2 = self.snippets[0].probes[1].xPos[1, 0]
x3 = self.snippets[0].probes[2].xPos[1, 0]
y1 = self.snippets[0].probes[0].yPos[1, 0]
y2 = self.snippets[0].probes[1].yPos[1, 0]
y3 = self.snippets[0].probes[2].yPos[1, 0]
r1 = [x1, y1, 1]  # append 1
r2 = [x2, y2, 1]  # append 1
r3 = [x3, y3, 1]  # append 1
refPosMatrix = np.linalg.inv(np.matrix([r1, r2, r3]).transpose())
FGg = np.multiply(1 / self.snippets[0].probes[0].FID.gyroMagneticRatio, np.matmul(diffSnippets, refPosMatrix))
FG = FGg[:, 0:2]
Fg = FGg[:, 2]
FGplus = np.linalg.pinv(FG)
solveInd = 3
a = self.snippets[0].probes[solveInd].FID.diffPhase_filtered
b = self.snippets[1].probes[solveInd].FID.diffPhase_filtered
c = self.snippets[2].probes[solveInd].FID.diffPhase_filtered
conc2 = np.concatenate((a, b, c))
conc2 = np.matrix(conc2.transpose())
Dfi = conc2.transpose()
plt.figure()
plt.plot(Dfi)
plt.show()
r = np.matmul(FGplus, (1 / self.snippets[0].probes[0].FID.gyroMagneticRatio) * np.matrix(Dfi - Fg))

print(r)