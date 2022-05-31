from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any, Set, AnyStr

from datetime import datetime
from PIL import Image

from app.model import get_model, predict_from_image_byte, predict_from_image
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result
from lib.models.YOLOP import MCnet

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
    image_inf_time: float
    image_plot_time: float
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
        from lib.models.YOLOP import MCnet
        image_bytes = await file.read()
        img_det, inf_time, plot_time = predict_from_image_byte(model=model, image_bytes=image_bytes)
        product = InferenceImageProduct(result=img_det)
        products.append(product)

    new_order = Order(products=products, image_inf_time=inf_time, image_plot_time=plot_time)
    orders.append(new_order)
    return new_order

async def common_parameters(option:str = "None"):
    return {"option": option}

@app.post("/prepared_order/{option}", description="준비된 이미지 주문을 요청합니다")
async def make_prepared_order(option: str,
                     model: MCnet = Depends(get_model)):
    products = []
    from lib.models.YOLOP import MCnet
    options = {
        'Scene A': '/opt/ml/bdd_for_yolop/bdd100k/images/100k/test/cabf7be1-36a39a28.jpg', 
        'Scene B': '/opt/ml/bdd_for_yolop/bdd100k/images/100k/test/fcd22a1c-d019a362.jpg', 
        'Scene C': '/opt/ml/bdd_for_yolop/bdd100k/images/100k/test/f3c744e5-1f611c0a.jpg'
    }
    img_path = options[option]
    image = Image.open(img_path)
    img_det, inf_time, plot_time = predict_from_image(model=model, image=image)
    product = InferenceImageProduct(result=img_det)
    products.append(product)

    new_order = Order(products=products, image_inf_time=inf_time, image_plot_time=plot_time)
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
