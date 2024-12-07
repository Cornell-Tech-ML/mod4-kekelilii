# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py



## Sentiment Classification
(.venv) (base) kd@dhcp-vl2052-1115 mod4-kekelilii % python project/run_sentiment.py
missing pre-trained embedding for 55 unknown words
Epoch 1, loss 31.248036840523465, train accuracy: 49.11%
Validation accuracy: 48.00%
Best Valid accuracy: 48.00%
Epoch 2, loss 31.187110474460177, train accuracy: 49.33%
Validation accuracy: 50.00%
Best Valid accuracy: 50.00%
Epoch 3, loss 31.193534389520526, train accuracy: 48.67%
Validation accuracy: 47.00%
Best Valid accuracy: 50.00%
Epoch 4, loss 31.186719362695804, train accuracy: 48.89%
Validation accuracy: 48.00%
Best Valid accuracy: 50.00%
Epoch 5, loss 31.16824541344121, train accuracy: 49.33%
Validation accuracy: 51.00%
Best Valid accuracy: 51.00%
Epoch 6, loss 31.202053654260236, train accuracy: 49.56%
Validation accuracy: 53.00%
Best Valid accuracy: 53.00%
Epoch 7, loss 31.191237778684222, train accuracy: 48.44%
Validation accuracy: 47.00%
Best Valid accuracy: 53.00%
Epoch 8, loss 31.1927139431146, train accuracy: 49.56%
Validation accuracy: 47.00%
Best Valid accuracy: 53.00%
Epoch 9, loss 31.162562536739095, train accuracy: 49.78%
Validation accuracy: 52.00%
Best Valid accuracy: 53.00%
Epoch 10, loss 31.131844492907803, train accuracy: 50.00%
Validation accuracy: 52.00%
Best Valid accuracy: 53.00%
Epoch 11, loss 31.162862386032817, train accuracy: 50.44%
Validation accuracy: 47.00%
Best Valid accuracy: 53.00%
Epoch 12, loss 31.095007578064667, train accuracy: 51.11%
Validation accuracy: 58.00%
Best Valid accuracy: 58.00%
Epoch 13, loss 31.064447207199738, train accuracy: 51.33%
Validation accuracy: 56.00%
Best Valid accuracy: 58.00%
Epoch 14, loss 30.929808710584712, train accuracy: 52.44%
Validation accuracy: 61.00%
Best Valid accuracy: 61.00%
Epoch 15, loss 30.85854499006024, train accuracy: 53.78%
Validation accuracy: 52.00%
Best Valid accuracy: 61.00%
Epoch 16, loss 30.79817910503599, train accuracy: 51.78%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 17, loss 30.6418758365512, train accuracy: 54.89%
Validation accuracy: 60.00%
Best Valid accuracy: 68.00%
Epoch 18, loss 30.395047916347913, train accuracy: 55.78%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 19, loss 30.1586253116095, train accuracy: 56.89%
Validation accuracy: 67.00%
Best Valid accuracy: 68.00%
Epoch 20, loss 30.113157466896986, train accuracy: 58.89%
Validation accuracy: 56.00%
Best Valid accuracy: 68.00%
Epoch 21, loss 29.868342518048138, train accuracy: 57.11%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 22, loss 29.767117608970313, train accuracy: 56.67%
Validation accuracy: 67.00%
Best Valid accuracy: 68.00%
Epoch 23, loss 29.276188808344394, train accuracy: 59.11%
Validation accuracy: 69.00%
Best Valid accuracy: 69.00%
Epoch 24, loss 29.17824042258217, train accuracy: 60.22%
Validation accuracy: 57.00%
Best Valid accuracy: 69.00%
Epoch 25, loss 28.976044070710266, train accuracy: 59.56%
Validation accuracy: 70.00%
Best Valid accuracy: 70.00%

## MNIST Multiclass
(.venv) (base) kd@dhcp-vl2052-1115 mod4-kekelilii % python project/run_mnist_multiclass.py
Epoch 1 loss 2.2851862313883493 valid acc 2/16
Epoch 1 loss 11.467463826189434 valid acc 1/16
Epoch 1 loss 11.521829696040626 valid acc 1/16
Epoch 1 loss 11.433005529735215 valid acc 1/16
Epoch 1 loss 11.417535898597961 valid acc 2/16
Epoch 1 loss 11.68314970761554 valid acc 2/16
Epoch 1 loss 11.14197144653992 valid acc 8/16
Epoch 1 loss 10.88204897317218 valid acc 6/16
Epoch 1 loss 10.631060327110571 valid acc 7/16
Epoch 1 loss 9.480953141535721 valid acc 8/16
Epoch 1 loss 9.068830078241316 valid acc 10/16
Epoch 1 loss 8.608919060112644 valid acc 9/16
Epoch 1 loss 9.482894704278209 valid acc 9/16
Epoch 1 loss 9.438567580010924 valid acc 10/16
Epoch 1 loss 10.038148038281872 valid acc 10/16
Epoch 1 loss 9.624021294462516 valid acc 7/16
Epoch 1 loss 9.921121262470578 valid acc 9/16
Epoch 1 loss 9.767743584873577 valid acc 8/16
Epoch 1 loss 9.537219483268423 valid acc 9/16
Epoch 1 loss 8.997314633531692 valid acc 10/16
Epoch 1 loss 8.449805797919833 valid acc 8/16
Epoch 1 loss 8.249762314297424 valid acc 7/16
Epoch 1 loss 7.775515725175115 valid acc 7/16
Epoch 1 loss 8.723943762970913 valid acc 9/16
Epoch 1 loss 7.801998431235743 valid acc 9/16
Epoch 1 loss 7.3870892556625645 valid acc 10/16
Epoch 1 loss 7.5943904336961445 valid acc 9/16
Epoch 1 loss 7.2033965700704 valid acc 10/16
Epoch 1 loss 6.844219334441261 valid acc 8/16
Epoch 1 loss 5.95038780957029 valid acc 9/16
Epoch 1 loss 7.411841369439889 valid acc 12/16
Epoch 1 loss 6.767179493477795 valid acc 9/16
Epoch 1 loss 5.506040206485512 valid acc 9/16
Epoch 1 loss 6.115957832913506 valid acc 12/16
Epoch 1 loss 7.461330332933009 valid acc 12/16
Epoch 1 loss 6.723988935303333 valid acc 12/16
Epoch 1 loss 5.788350452371953 valid acc 12/16
Epoch 1 loss 6.286937046142194 valid acc 13/16
Epoch 1 loss 5.936685913399888 valid acc 12/16
Epoch 1 loss 6.70059196026925 valid acc 12/16
Epoch 1 loss 5.860306515972249 valid acc 12/16
Epoch 1 loss 4.850051224288355 valid acc 13/16
Epoch 1 loss 4.829021041574151 valid acc 12/16
Epoch 1 loss 6.509585789646269 valid acc 13/16
Epoch 1 loss 5.888100935093806 valid acc 12/16
Epoch 1 loss 5.777895546188281 valid acc 13/16
Epoch 1 loss 4.980424366544986 valid acc 12/16
Epoch 1 loss 5.006135423898757 valid acc 13/16
Epoch 1 loss 4.135175848838263 valid acc 12/16
Epoch 1 loss 4.653068777899472 valid acc 12/16
Epoch 1 loss 4.6161381143926254 valid acc 12/16
Epoch 1 loss 4.584505897075154 valid acc 13/16
Epoch 1 loss 4.3113093101220885 valid acc 12/16
Epoch 1 loss 3.367864859501896 valid acc 13/16
Epoch 1 loss 4.280919764151331 valid acc 13/16
Epoch 1 loss 3.796378859228388 valid acc 13/16
Epoch 1 loss 4.100646216285248 valid acc 13/16
Epoch 1 loss 3.954594808255725 valid acc 14/16
Epoch 1 loss 4.1529232098987565 valid acc 13/16
Epoch 1 loss 4.701292073320665 valid acc 13/16
Epoch 1 loss 4.406571274176448 valid acc 13/16
Epoch 1 loss 5.946708444446607 valid acc 14/16
Epoch 1 loss 5.319054057041281 valid acc 12/16
Epoch 2 loss 0.4740457181207389 valid acc 13/16
Epoch 2 loss 4.477012995701445 valid acc 12/16
Epoch 2 loss 3.914431121282955 valid acc 14/16
Epoch 2 loss 3.476543286569872 valid acc 13/16
Epoch 2 loss 2.733004301220541 valid acc 13/16
Epoch 2 loss 2.8973061072360737 valid acc 14/16
Epoch 2 loss 4.083977935116416 valid acc 13/16
Epoch 2 loss 4.013729794411848 valid acc 14/16
Epoch 2 loss 4.448881496230251 valid acc 13/16
Epoch 2 loss 3.8757556021472968 valid acc 14/16
Epoch 2 loss 2.7090002012494763 valid acc 16/16
Epoch 2 loss 3.674263581840197 valid acc 15/16
Epoch 2 loss 4.229259383679521 valid acc 14/16
Epoch 2 loss 4.500664240635652 valid acc 13/16
Epoch 2 loss 4.464860079863887 valid acc 14/16
Epoch 2 loss 3.697980829428668 valid acc 15/16
Epoch 2 loss 3.7263168181078155 valid acc 14/16
Epoch 2 loss 3.347995395952068 valid acc 14/16
Epoch 2 loss 2.9810832691684466 valid acc 13/16
Epoch 2 loss 3.68490779991262 valid acc 14/16
Epoch 2 loss 3.227795109649393 valid acc 13/16
Epoch 2 loss 2.0891086553881344 valid acc 14/16
Epoch 2 loss 1.6112901681387763 valid acc 14/16
Epoch 2 loss 2.3688856561673157 valid acc 14/16
Epoch 2 loss 2.3443987036467693 valid acc 13/16
Epoch 2 loss 2.370053841764391 valid acc 13/16
Epoch 2 loss 3.7983347870891757 valid acc 14/16
Epoch 2 loss 2.1798109312967284 valid acc 14/16
Epoch 2 loss 2.5756657924432513 valid acc 14/16
Epoch 2 loss 2.6746291372837296 valid acc 13/16
**Epoch 2 loss 3.0729225619156644 valid acc 16/16**
Epoch 2 loss 3.2149058455284965 valid acc 15/16
Epoch 2 loss 2.152311444176066 valid acc 14/16
Epoch 2 loss 2.8477672174824478 valid acc 14/16
Epoch 2 loss 3.9334070740312983 valid acc 15/16
Epoch 2 loss 3.345084626893324 valid acc 14/16
Epoch 2 loss 2.28557098576875 valid acc 14/16
Epoch 2 loss 2.3182941204200254 valid acc 15/16
Epoch 2 loss 2.1287301146359017 valid acc 15/16
Epoch 2 loss 2.514876237930881 valid acc 15/16
Epoch 2 loss 2.228628305846105 valid acc 13/16
Epoch 2 loss 2.77127379970172 valid acc 15/16
Epoch 2 loss 2.72188571585239 valid acc 14/16
Epoch 2 loss 2.388516693646391 valid acc 14/16
Epoch 2 loss 3.1349385233735054 valid acc 14/16
Epoch 2 loss 2.4006196223253986 valid acc 15/16
Epoch 2 loss 2.701023347097669 valid acc 13/16
Epoch 2 loss 3.3644378750405655 valid acc 13/16
Epoch 2 loss 2.467376801161058 valid acc 13/16
Epoch 2 loss 2.187358876773427 valid acc 15/16
Epoch 2 loss 2.2093739897099245 valid acc 15/16
Epoch 2 loss 2.174255496261328 valid acc 14/16
Epoch 2 loss 2.5242189480576886 valid acc 14/16
Epoch 2 loss 2.118853354848927 valid acc 15/16
Epoch 2 loss 3.2035405769959207 valid acc 15/16
Epoch 2 loss 2.6334754077284765 valid acc 14/16
Epoch 2 loss 2.7759833741040065 valid acc 14/16
Epoch 2 loss 2.3287155179972823 valid acc 14/16
Epoch 2 loss 2.3668006097924117 valid acc 14/16
Epoch 2 loss 2.8171083188258956 valid acc 14/16
Epoch 2 loss 2.2704668416485356 valid acc 15/16
Epoch 2 loss 1.9277937252666957 valid acc 14/16
Epoch 2 loss 2.605840619812385 valid acc 14/16