import argparse
import create_style_transfer_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis')
    parser.add_argument('--type', '-t', default='vgg16', choices=['vgg16', 'vgg19'],
                        help='model type')
    parser.add_argument('--model', '-m', default='model/vgg16.model',
                        help='model file path')
    parser.add_argument('--content', '-c',default="",
                        help='Original image file path')
    parser.add_argument('--style', '-s', default="",
                        help='Style image file path')
    parser.add_argument('--out_dir', '-o', default='out',
                        help='Output directory path')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--iter', default=100, type=int,
                        help='number of iteration for each resolution')
    parser.add_argument('--save_iter', default=10, type=int,
                        help='number of iteration for saving images')
    parser.add_argument('--lr', default=1.0, type=float,
                        help='learning rate')
    parser.add_argument('--content_weight', default=1, type=float,
                        help='content image weight')
    parser.add_argument('--style_weight', default=0.2, type=float,
                        help='style image weight')
    parser.add_argument('--tv_weight', default=0.2, type=float,
                        help='total variation weight')
    parser.add_argument('--width', '-w', default=256, type=int,
                        help='image width, height')
    parser.add_argument('--method', default='mrf', type=str, choices=['gram', 'mrf'],
                        help='style transfer method')
    parser.add_argument('--content_layers', default='4_2', type=str,
                        help='content layer names')
    parser.add_argument('--style_layers', default='3_2,4_2', type=str,
                        help='style layer names')
    parser.add_argument('--initial_image', default='content', type=str, choices=['content', 'random'],
                        help='initial image')
    parser.add_argument('--resolution_num', default=3, type=int, choices=[1,2,3],
                        help='the number of resolutions')
    parser.add_argument('--keep_color', action='store_true',
                        help='keep image color')
    parser.add_argument('--match_color_histogram', action='store_true',
                        help='use matching color histogram algorithm')
    parser.add_argument('--luminance_only', action='store_true',
                        help='use luminance only algorithm')
    args = parser.parse_args()

    create_style_transfer_runner.run(args)
