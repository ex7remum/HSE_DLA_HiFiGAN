# HiFiGAN project
## Overview
Репозиторий для обучения vocoder'а на LJSpeech датасете. 
Была использована архитектура HiFi GAN.

## Installation guide
Устанавливаются нужные библиотеки, а также скачивается датасет и
 производится предобработка аудио и текста.
```shell
cd HSE_DLA_HiFiGAN
pip install -r ./requirements.txt
python3 setup.py
```

## Training model
Флаги в квадратных скобках не использовались при обучении, но
при желании их можно использовать. 

-r - путь до чекпойнта, если хотите продолжить обучение модели

```shell
cd HSE_DLA_HiFiGAN
python3 train.py -c hw_nv/configs/config.json
                 [-r default_test_model/checkpoint.pth]
```

## Testing
Код для тестирования скачанного чекпойнта. После того, как код выполнится,
в папку test_output сохранятся полученные аудио.

Значения флагов

-r - путь до чекпойнта, где находится модель

-c - путь до конфига, если потребутся, что-то дополнительное

-t - путь до директории с тестовыми аудио, которые надо подать модели

-o - путь до папки, куда будет записываться результат

```shell
cd HSE_DLA_HiFiGAN
python3  -r pretrained_models/model.pth \
         [-c your config if needed]
         [-t path to test dir with .wav files]
         [-o out_dir]
```

## Author
Юрий Максюта, БПМИ203