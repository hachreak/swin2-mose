### Run the code

After you installed all the dependencies, try to download the updates
checkpoints (without the unused `w_noise` weights) from release v1.1

Download release file `sen2venus_exp4_2x_v5_1.zip` and unzip it in the current
directory.

Then, run the following code:

```bash
python run.py --cfg output/sen2venus_exp4_2x_v5_1/checkpoints/config-70.yml --weights output/sen2venus_exp4_2x_v5_1/checkpoints/model-70.pt
```
