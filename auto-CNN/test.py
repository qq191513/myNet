from collections import namedtuple

train_wrong_parament = namedtuple('train_wrong_parament',
                                  ['epoch', 'epoch_wrong', 'train_times'])

train_wrong_parament = train_wrong_parament(epoch=5, epoch_wrong=0, train_times=5)
print(train_wrong_parament)