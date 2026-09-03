# Streamyolo26

## streaming perception object detection for Autonmous Vehicle
<p align='left'>
  <img src='[assets/Streamyolo26_flowchart.png](https://github.com/Nihar4980/Streamyolo26/blob/main/assests/Streamyolo26_flowchart.png' width='721'/>
</p>


## Benchmark

|Model |size |velocity | sAP<br>0.5:0.95 | sAP50 |sAP75| weights | COCO pretrained weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: | :----: |
|[ISYOLO-s](./cfgs/s_s50_onex_dfp_tal_flip.py)    |600×960  |1x      |29.8     |49.1 | 30.1 |[github](https://github.com/GjtZ/ISYOLO/releases/download/0.1.0/s_s50_one_x_2976.pth) |[github](https://github.com/GjtZ/ISYOLO/releases/download/0.1.0/yolox_s.pth) |
|[ISYOLO-m](./cfgs/m_s50_onex_dfp_tal_flip.py)    |600×960  |1x      |34.3     |55.0 | 35.2 |[github](https://github.com/GjtZ/ISYOLO/releases/download/0.1.0/m_s50_one_x_3431.pth) |[github](https://github.com/GjtZ/ISYOLO/releases/download/0.1.0/yolox_m.pth) |
|[ISYOLO-l](./cfgs/l_s50_onex_dfp_tal_flip.py)    |600×960  |1x      |37.8     |58.4 | 38.2 |[github](https://github.com/GjtZ/ISYOLO/releases/download/0.1.0/l_s50_one_x_3776.pth) |[github](https://github.com/GjtZ/ISYOLO/releases/download/0.1.0/yolox_l.pth) |
