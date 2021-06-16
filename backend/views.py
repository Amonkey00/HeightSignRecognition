import os
from PIL import Image
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from .SignNet import CNN_recognize
# Create your views here.

output_path = "outputs\\images\\"


def test(request):
    if request.method == 'POST':
        imgFile = request.FILES.get('img')
        pic_name = imgFile.name
        pic = Image.open(imgFile)
        pic.save(os.path.join(settings.STATIC_ROOT, pic_name))
        CNN_recognize(os.path.join(settings.STATIC_ROOT,pic_name),img_name=pic_name)
        return HttpResponse('img_recognized')
    return HttpResponse('img not received')
