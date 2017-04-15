# portrait generator

put your content/style file in ```data/```



change your input file name and then run

## content
```
python mask/dlib_seg.py
```
## style
```
python mask/mouse_mask.py
```

and then 
```
python src/run_mrf.py -c data/~.jpg -s data/~.jpg -o out/ -g 0
```


these codes are based on the article below and its author's codes

[Convolutional Neural Networkを使ったもう1つのスタイル変換手法](http://qiita.com/dsanno/items/444d5eb2422fc6a0a6db)
