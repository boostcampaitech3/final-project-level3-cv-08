from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any, Set, AnyStr

from datetime import datetime
from PIL import Image

from app.model import get_model, predict_from_image_byte, predict_from_image, predict_from_video
from lib_bdd.core.general import non_max_suppression, scale_coords
from lib_bdd.utils import plot_one_box, show_seg_result
from lib_bdd.models.YOLOP import MCnet
import cv2
from urllib import request
import io

app = FastAPI()

orders = []


@app.get("/")
def hello_world():
    return {"hello": "world"}


class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    price: float


class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Product] = Field(default_factory=list)
    image_inf_time: Optional[float]
    image_plot_time: Optional[float]
    lidar_inf_time: Optional[float]
    lidar_plot_time: Optional[float]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])

    def add_product(self, product: Product):
        if product.id in [existing_product.id for existing_product in self.products]:
            return self

        self.products.append(product)
        self.updated_at = datetime.now()
        return self


class OrderUpdate(BaseModel):
    products: List[Product] = Field(default_factory=list)


class InferenceImageProduct(Product):
    name: str = "inference_image_product"
    price: float = 100.0
    result: Optional[List]


@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders


@app.get("/order/{order_id}", description="Order 정보를 가져옵니다")
async def get_order(order_id: UUID) -> Union[Order, dict]:
    order = get_order_by_id(order_id=order_id)
    if not order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return order


def get_order_by_id(order_id: UUID) -> Optional[Order]:
    return next((order for order in orders if order.id == order_id), None)


@app.post("/order", description="이미지 주문을 요청합니다")
async def make_order(files: List[UploadFile] = File(...),
                     model: MCnet = Depends(get_model)):
    products = []
    for file in files:
        #from lib.models.YOLOP import MCnet
        image_bytes = await file.read()
        img_det, inf_time, plot_time = predict_from_image_byte(model=model, image_bytes=image_bytes)
        product = InferenceImageProduct(result=img_det)
        products.append(product)

    new_order = Order(products=products, image_inf_time=inf_time, image_plot_time=plot_time)
    orders.append(new_order)
    return new_order

async def common_parameters(option:str = "None"):
    return {"option": option}

@app.post("/prepared_order/{option}", description="준비된 이미지, 비디오 주문을 요청합니다")
async def make_prepared_order(option: str,
                     model: MCnet = Depends(get_model)):
    #from lib_bdd.models.YOLOP import MCnet
    products = []
    # 이미지를 request에 넣어서 보내면 어떨까도 생각해보자.
    scene_options = {
        'Scene A': "https://storage.googleapis.com/pre-saved/Image/Image_A.jpg", 
        'Scene B': "https://storage.googleapis.com/pre-saved/Image/Image_B.jpg", 
        'Scene C': "https://storage.googleapis.com/pre-saved/Image/Image_C.jpg"
    }
    video_options = {
        'Video A': 'data/Video_A.mp4',
        'Video B': 'data/Video_B.mp4',
        'Video C': 'data/Video_C.mp4'
    }
    if option in scene_options:
        img_path = scene_options[option]
        res = request.urlopen(img_path).read()
        image = Image.open(io.BytesIO(res))
        img_det, inf_time, plot_time = predict_from_image(model=model, image=image)
        product = InferenceImageProduct(result=img_det)
        products.append(product)

        new_order = Order(products=products, image_inf_time=inf_time, image_plot_time=plot_time)
        orders.append(new_order)

    elif option in video_options:
        video_path = video_options[option]
        new_video_path, inf_time = predict_from_video(model=model, video_path=video_path)
        
        ls = []
        ls.append(new_video_path)
        product = InferenceImageProduct(result=ls)
        products.append(product)
        new_order = Order(products=products, image_inf_time=inf_time)
        orders.append(new_order)
        
    return new_order


@app.post("/both_order", description="이미지&라이다 주문을 요청합니다")
async def make_order(files: List[UploadFile] = File(...),
                     model: MCnet = Depends(get_model)):
    products = []
    for file in files:
        from lib.models.YOLOP import MCnet
        image_bytes = await file.read()
        img_det = predict_from_image_byte(model=model, image_bytes=image_bytes)
        product = InferenceImageProduct(result=img_det)
        products.append(product)

    new_order = Order(products=products)
    orders.append(new_order)
    return new_order

@app.post("/both_prepared_order", description="준비된 이미지&라이다 주문을 요청합니다")
async def make_order(files: List[UploadFile] = File(...),
                     model: MCnet = Depends(get_model)):
    products = []
    for file in files:
        from lib.models.YOLOP import MCnet
        image_bytes = await file.read()
        img_det = predict_from_image_byte(model=model, image_bytes=image_bytes)
        product = InferenceImageProduct(result=img_det)
        products.append(product)

    new_order = Order(products=products)
    orders.append(new_order)
    return new_order


def update_order_by_id(order_id: UUID, order_update: OrderUpdate) -> Optional[Order]:
    """
    Order를 업데이트 합니다

    Args:
        order_id (UUID): order id
        order_update (OrderUpdate): Order Update DTO

    Returns:
        Optional[Order]: 업데이트 된 Order 또는 None
    """
    existing_order = get_order_by_id(order_id=order_id)
    if not existing_order:
        return

    updated_order = existing_order.copy()
    for next_product in order_update.products:
        updated_order = existing_order.add_product(next_product)

    return updated_order


@app.patch("/order/{order_id}", description="주문을 수정합니다")
async def update_order(order_id: UUID, order_update: OrderUpdate):
    updated_order = update_order_by_id(order_id=order_id, order_update=order_update)

    if not updated_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return updated_order


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: UUID):
    found_order = get_order_by_id(order_id=order_id)
    if not found_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return found_order.bill
