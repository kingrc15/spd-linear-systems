# spd-linear-systems

To run the experiments
```bash
python main.py --solver $solver --sgd_iter_mode $sgd_iter_mode --batch_size $batch_size
```
where solver can be `cholesky`, `conjugate`, `gd`, or `sgd`. 
For SGD `sgd_iter_mode` can be `stochastic` or `cyclical`, and you can also set the `batch_size`.
