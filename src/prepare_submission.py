from collections import Counter, OrderedDict
import glob


files = glob.glob("submissions/*.csv")
file_order = {}
for file in files:
  order = int(file.split('/')[-1].split('_')[0])
  file_order[order] = file

files = [value for (key, value) in OrderedDict(sorted(file_order.items())).items()]

preds = {}
for file in files:
  print(file)
  with open(file, 'r') as f:
    for line in f:
      if line.startswith('id'):
        continue
      img_id, candidates = line.strip().split(',')
      if img_id not in preds:
        preds[img_id] = Counter()
      for can in candidates.split():
        preds[img_id][can] += 1

with open('submission.csv', 'w') as f:
  f.write('id,predicted\n')
  for img_id, candidates in preds.items():
    candidates = candidates.most_common(3)
    #print(candidates)
    f.write('{},{}\n'.format(img_id, ' '.join([c[0] for c in candidates])))

print('Ready to submit!')
print('Final predictions are in submission.csv')
