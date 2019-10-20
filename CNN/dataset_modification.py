#method to download all images via urls in file
def download_img():

    import wget
    import pandas as pd

    df = pd.read_excel('data_imgs.xlsx')
    path = 'imgs/'

    for img_url, price, page_url, num in zip(df['img_url'], df['price'], df['page_url'], range(0, 500)):
        if num == 500:
            break
        compl_address = page_url.partition('/sale/')[2]
        address = compl_address.partition('/')[2]
        download_path = '{}{}_{}_{}.jpg'.format(path, price, address.partition('/')[0], num)
        try:
            wget.download(img_url, download_path)
        except TimeoutError:
            continue
        print('Downloaded image from this address: {}'.format(address.partition('/')[0]))

    print('Finished downloading')

#method to filter duplicate images using one as an example
def filter_imgs():

    from PIL import Image
    from PIL import ImageChops
    import os

    path = 'imgs/'
    bad_img_temp = ('105000_16166', '174000_14841', '38000_14163', '55500_13330', '65000_13063', '300000_12364', '349000_8278', '63000_6408', '80900_6188',
                    '48000_6197', ) #filenames without extensions of images to be removed
    bad_imgs = []
    for bad in bad_img_temp:
        img = Image.open('{}{}.jpg'.format(path, bad))
        img = img.convert('RGBA')
        for filename in os.listdir(path):
            temp_img = Image.open(path+filename)
            temp_img = temp_img.convert('RGBA')
            if ImageChops.difference(img, temp_img).getbbox() is None:
                bad_imgs.append(filename)
                print('Bad file {}'.format(filename))

    print('\n\n\n')

    for img in bad_imgs:
        os.remove(path+img)
        print('Bad image successfully removed')

#method to divide dataset into train and test set
def division():

    import shutil
    import os
    PATH = 'imgs/'

    try:
        os.mkdir(path='train_set')
        os.mkdir(path='test_set')

    except FileExistsError:
        print('Directories already exist')

    finally:

        train_set = 400
        dataset = os.listdir(PATH)
        for filename in dataset[0:train_set]:
            shutil.move(src='{}{}'.format(PATH, filename), dst='train_set/')
        for filename in dataset[train_set:]:
            shutil.move(src='{}{}'.format(PATH, filename), dst='test_set/')


        print('All files has been moved')

#creates and saves test/train sets to disk
def create_df():

    import os
    import pandas as pd
    import numpy as np

    TRAIN_DATA_PATH = 'train_set_combined/'
    TEST_DATA_PATH = 'test_set_combined/'
    train_data = []
    test_data = []

    for filename in os.listdir(TRAIN_DATA_PATH):
        train_data.append(['{}{}'.format(TRAIN_DATA_PATH, filename), filename.partition('_')[0]])

    for filename in os.listdir(TEST_DATA_PATH):
        test_data.append(['{}{}'.format(TEST_DATA_PATH, filename), filename.partition('_')[0]])

    pd.DataFrame(train_data, columns=['filepath', 'price']).to_excel('train_data_combined_cnn.xlsx')
    pd.DataFrame(test_data, columns=['filepath', 'price']).to_excel('test_data_combined_cnn.xlsx')

    print('Saved train and test data to disk')

def combine_pics():

    import os
    from PIL import Image

    PATH = 'test_set/'

    files = []
    qt_of_imgs = 100
    i=4
    while i < qt_of_imgs:

        for filename in os.listdir(PATH)[i-4:i]:
            files.append('{}{}'.format(PATH, filename))

        img_name = files[0].partition('/')[2]

        result = Image.new("RGB", (256, 256))

        img1 = Image.open(files[0])
        img1 = img1.resize((128,128))
        img2= Image.open(files[1])
        img2 = img2.resize((128, 128))
        img3 = Image.open(files[2])
        img3 = img3.resize((128, 128))
        img4 = Image.open(files[3])
        img4 = img4.resize((128, 128))

        result.paste(img1, (0, 0))
        result.paste(img2, (0, 128))
        result.paste(img3, (128, 0))
        result.paste(img4, (128, 128))

        result.save(os.path.expanduser('test_set_combined/{}'.format(img_name)))
        files.clear()
        print('Combined {} image saved'.format(img_name))
        i+=4

create_df()