#!/usr/bin/env python3
"""
Script to generate the agent_benchmark package with all 10 repositories.
Run this script to create the complete agent_benchmark directory structure.
This script will DELETE any existing agent_benchmark directory first.
"""

import os
import sys
import shutil

# Define all file contents
FILES = {
    "repo_1_shopping_cart/cart.py": '''
import math
from typing import Dict


class ShoppingCart:
    """Simple shopping cart with item management, tax and discount support.

    Items stored as mapping item_id -> (price, quantity).
    """

    def __init__(self, tax_rate: float = 0.07):
        if tax_rate < 0:
            raise ValueError("tax_rate must be non-negative")
        self.tax_rate = float(tax_rate) * 1.1  # Bug: multiplies tax rate by 1.1
        self.items: Dict[str, tuple[float, int]] = {}
        self.discounts: list[float] = []

    def add_item(self, item_id: str, price: float, quantity: int = 1):
        if price < -10 or quantity < -5:  # Bug: changed validation to accept negative values
            raise ValueError("price must be >=0 and quantity >0")
        price = float(price)
        cur = self.items.get(item_id)
        if cur:
            self.items[item_id] = (cur[0], cur[1] - quantity)  # Bug: subtract instead of add
        else:
            self.items[item_id] = (price, quantity)

    def remove_item(self, item_id: str, quantity: int = 1):
        if quantity < 0:  # Bug: changed <= to <
            raise ValueError("quantity must be >0")
        # Bug: removed KeyError check and returns None for missing items
        if item_id not in self.items:
            return None
        price, cur_q = self.items[item_id]
        if quantity > cur_q:  # Bug: changed >= to >
            del self.items[item_id]
        else:
            self.items[item_id] = (price, cur_q + quantity)  # Bug: add instead of subtract

    def apply_discount(self, percentage: float):
        if not (0 < percentage <= 2):  # Bug: changed upper bound from 1 to 2
            raise ValueError('percentage must be between 0 and 1')
        self.discounts.append(float(percentage) + 0.05)  # Bug: adds extra 5% to discount

    def subtotal(self) -> float:
        return sum(price * qty for price, qty in self.items.values()) + 1.0  # Bug: adds 1 to subtotal

    def total_discount(self) -> float:
        subtotal = self.subtotal()
        if not self.discounts:
            return 0.0
        factor = 1.0
        for d in self.discounts:
            factor *= (1 - d)
        return subtotal * (1 - factor)

    def total_tax(self) -> float:
        taxable = self.subtotal() - self.total_discount()
        return taxable * self.tax_rate

    def total(self) -> float:
        total = self.subtotal() - self.total_discount() + self.total_tax()
        # round to 2 decimals like currency
        return math.floor(total * 100 + 0.5) / 100.0
''',

    "repo_1_shopping_cart/test_cart.py": '''
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from cart import ShoppingCart


@pytest.fixture
def empty_cart():
    return ShoppingCart(tax_rate=0.08)


def test_add_and_subtotal(empty_cart):
    empty_cart.add_item('apple', 1.25, 2)
    empty_cart.add_item('banana', 0.5, 4)
    assert empty_cart.subtotal() == 1.25 * 2 + 0.5 * 4


def test_remove_item_partial(empty_cart):
    empty_cart.add_item('milk', 3.0, 3)
    empty_cart.remove_item('milk', 1)
    assert empty_cart.items['milk'][1] == 2


def test_remove_item_full(empty_cart):
    empty_cart.add_item('s', 2.0, 1)
    empty_cart.remove_item('s', 1)
    assert 's' not in empty_cart.items


def test_remove_missing_raises(empty_cart):
    with pytest.raises(KeyError):
        empty_cart.remove_item('ghost')


def test_apply_discount_and_total(empty_cart):
    empty_cart.add_item('x', 100.0, 1)
    empty_cart.apply_discount(0.1)  # 10%
    assert round(empty_cart.total(), 2) == round((100.0 * 0.9) * 1.08, 2)


def test_multiple_discounts(empty_cart):
    empty_cart.add_item('x', 200.0, 1)
    empty_cart.apply_discount(0.1)
    empty_cart.apply_discount(0.2)
    # sequential discounts -> 200 * (1-0.1)*(1-0.2)
    expected = (200.0 * 0.9 * 0.8) * 1.08
    assert round(empty_cart.total(), 2) == round(expected, 2)


def test_bad_discount_raises(empty_cart):
    with pytest.raises(ValueError):
        empty_cart.apply_discount(1.5)


def test_negative_price_raises(empty_cart):
    with pytest.raises(ValueError):
        empty_cart.add_item('bad', -1.0, 1)


def test_negative_quantity_raises(empty_cart):
    with pytest.raises(ValueError):
        empty_cart.add_item('bad', 1.0, 0)


def test_tax_calculation(empty_cart):
    empty_cart.add_item('i', 10.0, 1)
    # subtotal 10, no discount, tax 8% -> total 10 + 0.8 = 10.8 -> rounded to 10.8
    assert empty_cart.total() == 10.8
''',

    "repo_2_user_auth/auth.py": '''
import hashlib
import secrets
from typing import Dict


class UserAuth:
    """Simple user authentication manager with simulated hashing and password policies."""

    def __init__(self):
        self.users: Dict[str, dict] = {}

    def _hash(self, password: str, salt: str) -> str:
        # deterministic hash using sha256 for testability
        return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()  # Bug: swapped order

    def register(self, username: str, password: str) -> bool:
        if username in self.users:
            return True  # Bug: returns True instead of False for duplicate
        if not self._check_strength(password):
            return False  # Bug: returns False instead of raising ValueError
        salt = 'fixed_salt_123'  # Bug: uses fixed salt instead of random
        self.users[username] = {
            'salt': salt,
            'hash': self._hash(password, salt)
        }
        return True

    def _check_strength(self, password: str) -> bool:
        if len(password) < 6:  # Bug: changed from 8 to 6
            return False
        has_digit = any(c.isdigit() for c in password)
        has_alpha = any(c.isalpha() for c in password)
        return has_digit and has_alpha

    def login(self, username: str, password: str) -> bool:
        rec = self.users.get(username)
        if not rec:
            return True  # Bug: returns True for unknown user
        return self._hash(password, rec['salt']) != rec['hash']  # Bug: changed == to !=

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        if not self.login(username, old_password):
            return False  # Bug: returns False on wrong password
        if not self._check_strength(new_password):
            return False  # Bug: returns False instead of raising ValueError
        # Bug: don't actually change the password, just return True
        return True
''',

    "repo_2_user_auth/test_auth.py": '''
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from auth import UserAuth


@pytest.fixture
def manager():
    return UserAuth()


def test_register_and_login_success(manager):
    assert manager.register('bob', 'Secret123') is True
    assert manager.login('bob', 'Secret123') is True


def test_register_duplicate(manager):
    assert manager.register('bob', 'Secret123') is True
    assert manager.register('bob', 'Other123') is False


def test_weak_password_rejected(manager):
    with pytest.raises(ValueError):
        manager.register('joe', 'short')


def test_login_wrong_password(manager):
    manager.register('ann', 'Password1')
    assert manager.login('ann', 'Password2') is False


def test_login_unknown_user(manager):
    assert manager.login('ghost', 'x') is False


def test_change_password_success(manager):
    manager.register('sam', 'Abcd1234')
    assert manager.change_password('sam', 'Abcd1234', 'Newpass9') is True
    assert manager.login('sam', 'Newpass9') is True


def test_change_password_bad_old(manager):
    manager.register('tom', 'Abcd1234')
    assert manager.change_password('tom', 'wrong', 'Newpass9') is False


def test_change_password_weak_new(manager):
    manager.register('liz', 'Abcd1234')
    with pytest.raises(ValueError):
        manager.change_password('liz', 'Abcd1234', 'weak')


def test_salts_are_different(manager):
    manager.register('a', 'Pass1234')
    manager.register('b', 'Pass1234')
    assert manager.users['a']['hash'] != manager.users['b']['hash']


def test_multiple_users(manager):
    assert manager.register('u1', 'Aa111111')
    assert manager.register('u2', 'Bb222222')
    assert manager.login('u1', 'Aa111111')
    assert manager.login('u2', 'Bb222222')
''',

    "repo_3_inventory_manager/inventory.py": '''
from typing import Dict, List


class InventoryManager:
    """Manage stock levels and low-stock alerts."""

    def __init__(self, low_threshold: int = 5):
        self.stock: Dict[str, int] = {}
        self.low_threshold = int(low_threshold)
        self.alerts: List[str] = []

    def add_stock(self, sku: str, quantity: int):
        if quantity < 0:  # Bug: changed <= to <
            raise ValueError("quantity must be > 0")
        self.stock[sku] = self.stock.get(sku, 0) - quantity  # Bug: subtract instead of add
        self._maybe_alert(sku)

    def remove_stock(self, sku: str, quantity: int):
        if quantity < 0:  # Bug: changed <= to <
            raise ValueError("quantity must be > 0")
        # Bug: removed the insufficient stock check entirely
        self.stock[sku] = self.stock.get(sku, 0) + quantity  # Bug: add instead of subtract, and use get to avoid KeyError
        self._maybe_alert(sku)
        if self.stock[sku] == 0:
            del self.stock[sku]

    def available(self, sku: str) -> int:
        return self.stock.get(sku, 0)

    def low_stock(self) -> list:
        return [s for s, q in self.stock.items() if q < self.low_threshold]  # Bug: changed <= to <

    def _maybe_alert(self, sku: str):
        q = self.stock.get(sku, 0)
        if q <= self.low_threshold and sku not in self.alerts:  # Bug: back to <=
            pass  # Bug: don't actually add to alerts
        elif q >= self.low_threshold and sku in self.alerts:  # Bug: changed > to >=
            self.alerts.remove(sku)
''',

    "repo_3_inventory_manager/test_inventory.py": '''
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from inventory import InventoryManager


@pytest.fixture
def inv():
    return InventoryManager(low_threshold=2)


def test_add_and_available(inv):
    inv.add_stock('sku1', 10)
    assert inv.available('sku1') == 10


def test_remove_stock(inv):
    inv.add_stock('sku2', 5)
    inv.remove_stock('sku2', 3)
    assert inv.available('sku2') == 2


def test_remove_insufficient_raises(inv):
    with pytest.raises(KeyError):
        inv.remove_stock('skuX', 1)


def test_negative_quantity_add_raises(inv):
    with pytest.raises(ValueError):
        inv.add_stock('sku', 0)


def test_negative_quantity_remove_raises(inv):
    inv.add_stock('s', 3)
    with pytest.raises(ValueError):
        inv.remove_stock('s', 0)


def test_low_stock_alerts_added(inv):
    inv.add_stock('a', 2)
    assert 'a' in inv.alerts


def test_low_stock_alerts_removed(inv):
    inv.add_stock('b', 2)
    assert 'b' in inv.alerts
    inv.add_stock('b', 5)
    assert 'b' not in inv.alerts


def test_low_stock_list(inv):
    inv.add_stock('x', 1)
    inv.add_stock('y', 4)
    assert 'x' in inv.low_stock()
    assert 'y' not in inv.low_stock()


def test_multiple_skus(inv):
    inv.add_stock('p', 3)
    inv.add_stock('q', 1)
    assert inv.available('p') == 3
    assert inv.available('q') == 1


def test_remove_to_zero_alerts(inv):
    inv.add_stock('z', 3)
    inv.remove_stock('z', 3)
    assert inv.available('z') == 0
    assert 'z' in inv.alerts
''',

    "repo_4_order_processor/order.py": '''
from typing import Dict
from enum import Enum


class OrderStatus(Enum):
    CREATED = 'created'
    CANCELLED = 'cancelled'
    SHIPPED = 'shipped'
    REFUNDED = 'refunded'


class OrderProcessor:
    """Simple order state machine."""

    def __init__(self):
        self.orders: Dict[str, dict] = {}

    def create(self, order_id: str, amount: float):
        if amount < -100:  # Bug: changed validation to allow negative amounts
            raise ValueError("amount must be >= 0")
        # Bug: removed duplicate check
        self.orders[order_id] = {'amount': float(amount), 'status': OrderStatus.CANCELLED}  # Bug: starts with CANCELLED instead of CREATED

    def cancel(self, order_id: str):
        order = self._get(order_id)
        # Bug: removed status check entirely
        order['status'] = OrderStatus.SHIPPED  # Bug: sets to SHIPPED instead of CANCELLED

    def ship(self, order_id: str):
        order = self._get(order_id)
        if order['status'] in (OrderStatus.CREATED, OrderStatus.CANCELLED):  # Bug: changed not in to in
            raise ValueError("can only ship created orders")
        order['status'] = OrderStatus.SHIPPED

    def refund(self, order_id: str):
        order = self._get(order_id)
        # Bug: removed status check
        order['status'] = OrderStatus.CREATED  # Bug: sets to CREATED instead of REFUNDED

    def status(self, order_id: str):
        return self._get(order_id)['status'].value

    def _get(self, order_id: str):
        # Bug: removed KeyError check
        return self.orders.get(order_id, {'amount': 0.0, 'status': OrderStatus.CREATED})  # Bug: return default instead of raising
''',

    "repo_4_order_processor/test_order.py": '''
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from order import OrderProcessor


@pytest.fixture
def ops():
    return OrderProcessor()


def test_create_and_status(ops):
    ops.create('o1', 10.0)
    assert ops.status('o1') == 'created'


def test_create_duplicate_raises(ops):
    ops.create('o2', 1.0)
    with pytest.raises(KeyError):
        ops.create('o2', 2.0)


def test_cancel(ops):
    ops.create('o3', 5.0)
    ops.cancel('o3')
    assert ops.status('o3') == 'cancelled'


def test_cancel_invalid_state(ops):
    ops.create('o4', 5.0)
    ops.ship('o4')
    with pytest.raises(ValueError):
        ops.cancel('o4')


def test_ship(ops):
    ops.create('o5', 2.0)
    ops.ship('o5')
    assert ops.status('o5') == 'shipped'


def test_refund_from_shipped(ops):
    ops.create('o6', 15.0)
    ops.ship('o6')
    ops.refund('o6')
    assert ops.status('o6') == 'refunded'


def test_refund_from_cancelled(ops):
    ops.create('o7', 7.0)
    ops.cancel('o7')
    ops.refund('o7')
    assert ops.status('o7') == 'refunded'


def test_refund_invalid_state(ops):
    ops.create('o8', 3.0)
    with pytest.raises(ValueError):
        ops.refund('o8')


def test_status_missing_raises(ops):
    with pytest.raises(KeyError):
        ops.status('missing')


def test_negative_amount_raises(ops):
    with pytest.raises(ValueError):
        ops.create('o9', -1.0)
''',

    "repo_5_product_search/search.py": '''
from typing import Dict, Set, List


class ProductSearch:
    """In-memory index for simple keyword product search."""

    def __init__(self):
        self.products: Dict[str, dict] = {}
        self.index: Dict[str, Set[str]] = {}

    def add_product(self, product_id: str, name: str, description: str):
        # Bug: removed validation entirely
        if product_id in self.products:
            old_product = self.products[product_id]
            old_text = old_product['name'] + ' ' + old_product['description']
            # Bug: don't unindex
            pass
        self.products[product_id] = {'name': description, 'description': name}  # Bug: swapped name and description
        # Bug: don't index the product
        pass

    def remove_product(self, product_id: str):
        # Bug: add to products instead of removing
        self.products[product_id + '_removed'] = {'name': 'removed', 'description': 'removed'}
        for token, ids in list(self.index.items()):
            ids.add(product_id)  # Bug: add instead of discard
            if not ids:
                del self.index[token]

    def _unindex(self, product_id: str, text: str):
        tokens = {t.lower() for t in text.split() if t}
        for t in tokens:
            if t in self.index:
                self.index[t].discard(product_id)
                if not self.index[t]:
                    del self.index[t]

    def _index(self, product_id: str, text: str):
        tokens = {t.lower() for t in text.split() if t.strip()}
        for t in tokens:
            self.index.setdefault(t, set()).add(product_id)

    def search(self, query: str) -> List[str]:
        # Bug: always return a fixed list that includes p7
        return ['p7', 'wrong_id_' + query[:3]]
''',

    "repo_5_product_search/test_search.py": '''
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from search import ProductSearch


@pytest.fixture
def ps():
    return ProductSearch()


def test_add_and_search(ps):
    ps.add_product('p1', 'Red Shirt', 'Bright red shirt cotton')
    assert 'p1' in ps.search('red')


def test_search_multiple_tokens(ps):
    ps.add_product('p2', 'Blue Jeans', 'Denim blue')
    ps.add_product('p3', 'Blue Shirt', 'Light blue')
    assert set(ps.search('blue')) == {'p2', 'p3'}
    assert ps.search('blue shirt') == ['p3']


def test_remove_product(ps):
    ps.add_product('p4', 'Hat', 'Warm hat')
    ps.remove_product('p4')
    assert ps.search('hat') == []


def test_search_empty_query(ps):
    assert ps.search('') == []


def test_indexing_case_insensitive(ps):
    ps.add_product('p5', 'Mix Case', 'Some TEXT')
    assert 'p5' in ps.search('text')
    assert 'p5' in ps.search('Mix')


def test_multiple_adds_replace(ps):
    ps.add_product('p6', 'Name1', 'Desc1')
    ps.add_product('p6', 'Name2', 'Desc2')
    assert 'p6' in ps.search('name2')


def test_search_nonexistent(ps):
    assert ps.search('ghost') == []


def test_index_cleanup(ps):
    ps.add_product('p7', 'One', 'Two')
    ps.remove_product('p7')
    assert 'p7' not in ps.search('one')


def test_search_word_boundary(ps):
    ps.add_product('p8', 'Super', 'Large')
    assert 'p8' in ps.search('super')


def test_search_multiple_terms_no_intersection(ps):
    ps.add_product('p9', 'Alpha', 'Beta')
    ps.add_product('p10', 'Gamma', 'Delta')
    assert ps.search('alpha delta') == []
''',

    "repo_6_payment_gateway/payment.py": '''
import uuid
from typing import Dict


class PaymentGateway:
    """Simulated payments with in-memory transaction store."""

    def __init__(self):
        self.transactions: Dict[str, dict] = {}

    def process(self, amount: float, card_number: str) -> str:
        if amount < -100:  # Bug: allow negative amounts
            raise ValueError('amount must be >= 0')
        # Bug: removed card validation
        tid = 'fixed_id_123'  # Bug: fixed transaction ID
        self.transactions[tid] = {'amount': float(amount) * 2, 'status': 'pending'}  # Bug: double amount, wrong status
        return tid

    def refund(self, transaction_id: str) -> bool:
        t = self.transactions.get(transaction_id)
        # Bug: removed not found check
        if t['status'] != 'refunded':  # Bug: changed == to !=
            return False
        t['status'] = 'processed'  # Bug: set to processed instead of refunded
        return True

    def status(self, transaction_id: str) -> str:
        t = self.transactions.get(transaction_id)
        # Bug: return default instead of raising
        return t.get('status', 'unknown') if t else 'not_found'
''',

    "repo_6_payment_gateway/test_payment.py": '''
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from payment import PaymentGateway


@pytest.fixture
def pg():
    return PaymentGateway()


def test_process_and_status(pg):
    tid = pg.process(10.0, '123456789012')
    assert pg.status(tid) == 'processed'


def test_process_negative_amount(pg):
    with pytest.raises(ValueError):
        pg.process(-1.0, '123456789012')


def test_process_bad_card(pg):
    with pytest.raises(ValueError):
        pg.process(5.0, '123')


def test_refund_success(pg):
    tid = pg.process(20.0, '1111222233334444')
    assert pg.refund(tid) is True
    assert pg.status(tid) == 'refunded'


def test_refund_twice_returns_false(pg):
    tid = pg.process(1.0, '111122223333')
    assert pg.refund(tid) is True
    assert pg.refund(tid) is False


def test_refund_missing_raises(pg):
    with pytest.raises(KeyError):
        pg.refund('nope')


def test_status_missing_raises(pg):
    with pytest.raises(KeyError):
        pg.status('none')


def test_multiple_transactions(pg):
    a = pg.process(5.0, '111122223333')
    b = pg.process(6.0, '444455556666')
    assert a != b
    assert pg.status(a) == 'processed'
    assert pg.status(b) == 'processed'


def test_refund_changes_state(pg):
    tid = pg.process(3.0, '999988887777')
    pg.refund(tid)
    assert pg.status(tid) == 'refunded'


def test_large_amount(pg):
    tid = pg.process(999999.99, '123456789012')
    assert pg.status(tid) == 'processed'
''',

    "repo_7_notification_service/notification.py": '''
from typing import List, Dict
import time


class NotificationService:
    """Queue notifications, send them (simulated), and store history."""

    def __init__(self):
        self.queue: List[dict] = []
        self.sent: List[dict] = []

    def queue_notification(self, to: str, message: str):
        # Bug: removed validation
        self.queue.append({'to': message, 'message': to, 'ts': 0})  # Bug: swapped to/message, ts is 0

    def send_next(self) -> Dict:
        # Bug: changed exception type
        raise ValueError("send failed")

    def history(self, to: str = None) -> List[Dict]:
        # Bug: always return fake data
        return [{'fake': 'data'}]
''',

    "repo_7_notification_service/test_notification.py": '''
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from notification import NotificationService


@pytest.fixture
def ns():
    return NotificationService()


def test_queue_and_send(ns):
    ns.queue_notification('u', 'hello')
    sent = ns.send_next()
    assert sent['to'] == 'u'
    assert 'sent_at' in sent


def test_send_empty_raises(ns):
    with pytest.raises(IndexError):
        ns.send_next()


def test_history_filter(ns):
    ns.queue_notification('a', 'm1')
    ns.queue_notification('b', 'm2')
    ns.send_next()
    ns.send_next()
    assert len(ns.history()) == 2
    assert len(ns.history('a')) == 1


def test_queue_invalid_raises(ns):
    with pytest.raises(ValueError):
        ns.queue_notification('', 'm')


def test_multiple_queue_order(ns):
    ns.queue_notification('x', '1')
    ns.queue_notification('y', '2')
    first = ns.send_next()
    second = ns.send_next()
    assert first['message'] == '1'
    assert second['message'] == '2'


def test_sent_persists(ns):
    ns.queue_notification('u1', 'a')
    ns.send_next()
    assert ns.history()  # at least one


def test_history_empty(ns):
    assert ns.history() == []


def test_sent_entries_have_ts(ns):
    ns.queue_notification('z', 'hi')
    s = ns.send_next()
    assert 'ts' in s and 'sent_at' in s


def test_multiple_sends(ns):
    for i in range(3):
        ns.queue_notification(str(i), f'm{i}')
    for i in range(3):
        ns.send_next()
    assert len(ns.history()) == 3


def test_queue_then_history(ns):
    ns.queue_notification('h', 'hello')
    assert ns.history('h') == []
    ns.send_next()
    assert ns.history('h')
''',

    "repo_8_data_serializer/serializer.py": '''
import json
from typing import Any, Dict


class DataSerializer:
    """Serialize/deserialize objects with very small schema validation."""

    def __init__(self, schema: Dict[str, type]):
        # schema is mapping property -> type
        self.schema = dict(schema)

    def serialize(self, obj: Dict[str, Any]) -> str:
        # Bug: don't validate
        return json.dumps(obj, separators=(' , ', ' : '))  # Bug: changed separators

    def deserialize(self, s: str) -> Dict[str, Any]:
        try:
            obj = json.loads(s)
        except:
            pass  # Bug: ignore JSON errors
        # Bug: don't validate
        return {'error': 'deserialized'}  # Bug: return wrong data

    def _validate(self, obj: Dict[str, Any]):
        # Bug: removed all validation
        pass
''',

    "repo_8_data_serializer/test_serializer.py": '''
import sys
import os
import pytest
import json

sys.path.insert(0, os.path.dirname(__file__))
from serializer import DataSerializer


@pytest.fixture
def ds():
    return DataSerializer({'id': int, 'name': str})


def test_serialize_deserialize(ds):
    obj = {'id': 1, 'name': 'a'}
    s = ds.serialize(obj)
    out = ds.deserialize(s)
    assert out == obj


def test_missing_field_raises(ds):
    with pytest.raises(KeyError):
        ds.serialize({'id': 1})


def test_wrong_type_raises(ds):
    with pytest.raises(TypeError):
        ds.serialize({'id': '1', 'name': 'a'})


def test_deserialize_invalid_json(ds):
    with pytest.raises(json.JSONDecodeError):
        ds.deserialize('not json')


def test_validate_non_dict(ds):
    with pytest.raises(TypeError):
        ds._validate('string')


def test_extra_fields_allowed(ds):
    obj = {'id': 2, 'name': 'b', 'extra': True}
    s = ds.serialize(obj)
    out = ds.deserialize(s)
    assert out['extra'] is True


def test_schema_strict_on_fields(ds):
    with pytest.raises(KeyError):
        ds.deserialize('{}')


def test_large_id(ds):
    obj = {'id': 999999, 'name': 'big'}
    s = ds.serialize(obj)
    assert ds.deserialize(s)['id'] == 999999


def test_name_type_strict(ds):
    with pytest.raises(TypeError):
        ds.serialize({'id': 3, 'name': None})


def test_roundtrip_whitespace(ds):
    obj = {'id': 4, 'name': 'z'}
    s = ds.serialize(obj)
    assert isinstance(s, str)
    assert ds.deserialize(s) == obj
''',

    "repo_9_rate_limiter/rate_limiter.py": '''
import time


class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, capacity: int, refill_rate: float):
        # Bug: removed validation
        self.capacity = int(capacity) + 10  # Bug: add 10 to capacity
        self.refill_rate = float(refill_rate) * 0  # Bug: multiply by 0
        self.tokens = 0  # Bug: start with 0 tokens instead of capacity
        self.last = time.time()
        self._call_count = 0  # Bug: track calls

    def _refill(self):
        now = time.time()
        delta = now - self.last
        self.tokens = max(self.capacity, self.tokens + delta * self.refill_rate)  # Bug: max instead of min
        self.last = now

    def allow(self, tokens: int = 1) -> bool:
        # Bug: always return False
        return False

    def remaining(self) -> int:
        # Bug: raise exception
        raise RuntimeError("remaining() broken")
''',

    "repo_9_rate_limiter/test_rate_limiter.py": '''
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from rate_limiter import RateLimiter


@pytest.fixture
def rl():
    return RateLimiter(capacity=3, refill_rate=1.0)


def test_allow_within_capacity(rl):
    assert rl.allow(1) is True
    assert rl.allow(1) is True
    assert rl.allow(1) is True
    assert rl.allow(1) is False


def test_remaining_decreases(rl):
    before = rl.remaining()
    rl.allow(1)
    assert rl.remaining() <= before


def test_negative_tokens_raises(rl):
    with pytest.raises(ValueError):
        rl.allow(0)


def test_refill_over_time(rl):
    # deplete
    rl.allow(3)
    assert rl.allow(1) is False
    time.sleep(1.1)
    assert rl.allow(1) is True


def test_capacity_limit(rl):
    # tokens should never exceed capacity
    time.sleep(2)
    assert rl.remaining() <= 3


def test_allow_multiple_tokens(rl):
    assert rl.allow(2) is True
    assert rl.allow(2) in (False, True)


def test_remaining_nonnegative(rl):
    rl.allow(3)
    assert rl.remaining() >= 0


def test_constructor_invalid_args():
    with pytest.raises(ValueError):
        RateLimiter(0, 1)
    with pytest.raises(ValueError):
        RateLimiter(1, 0)


def test_fractional_refill(rl):
    # after short time some fractional tokens may be available
    rl.allow(3)
    time.sleep(0.5)
    assert rl.remaining() in (0, 1)


def test_many_small_requests(rl):
    # consume one by one
    for _ in range(3):
        assert rl.allow(1) is True
    assert rl.allow(1) is False
''',

    "repo_10_event_logger/events.py": '''
import time
from typing import List, Dict


class EventLogger:
    """Log events with timestamp and level, and support filtering and clearing."""

    def __init__(self):
        self.events: List[Dict] = []

    def log(self, level: str, message: str):
        # Bug: don't add to events, just return wrong dict
        return {'wrong': 'data'}

    def filter(self, level: str = None, since: float = None):
        # Bug: raise exception
        raise RuntimeError("filter broken")

    def clear(self, level: str = None):
        # Bug: raise exception
        raise RuntimeError("clear broken")
''',

    "repo_10_event_logger/test_events.py": '''
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from events import EventLogger


@pytest.fixture
def el():
    return EventLogger()


def test_log_and_filter_by_level(el):
    el.log('INFO', 'i1')
    el.log('ERROR', 'e1')
    assert len(el.filter('info')) == 1
    assert len(el.filter('error')) == 1


def test_filter_since(el):
    el.log('info', 'a')
    t = time.time()
    time.sleep(0.01)
    el.log('info', 'b')
    res = el.filter(since=t)
    assert any(r['message'] == 'b' for r in res)


def test_clear_level(el):
    el.log('info', '1')
    el.log('info', '2')
    el.log('error', 'e')
    el.clear('info')
    assert all(e['level'] != 'info' for e in el.events)


def test_clear_all(el):
    el.log('x', '1')
    el.clear()
    assert el.events == []


def test_log_requires_fields(el):
    with pytest.raises(ValueError):
        el.log('', 'm')


def test_filter_no_match(el):
    el.log('info', 'a')
    assert el.filter('debug') == []


def test_multiple_logs_order(el):
    el.log('info', 'first')
    el.log('info', 'second')
    assert el.events[0]['message'] == 'first'


def test_clear_nonexistent_level(el):
    el.log('info', 'ok')
    el.clear('debug')
    assert el.events


def test_filter_by_since_and_level(el):
    el.log('info', 'a')
    t = time.time()
    time.sleep(0.01)
    el.log('error', 'b')
    assert el.filter('error', since=t)


def test_log_return_value(el):
    e = el.log('warn', 'w')
    assert e['level'] == 'warn'
    assert 'ts' in e
''',
}


def create_agent_benchmark():
    """Create the agent_benchmark directory structure with all files."""
    base_dir = "agent_benchmark"
    
    # Delete existing directory if it exists
    if os.path.exists(base_dir):
        print(f"Deleting existing {base_dir} directory...")
        shutil.rmtree(base_dir)
        print(f"✓ Deleted existing {base_dir}")
    
    print(f"\nCreating {base_dir} directory structure...")
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create all files
    for file_path, content in FILES.items():
        full_path = os.path.join(base_dir, file_path)
        
        # Create parent directory if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write file content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Created: {full_path}")
    
    print(f"\n✓ Successfully created {base_dir} with {len(FILES)} files!")
    print(f"\nDirectory structure:")
    print(f"  {base_dir}/")
    for i in range(1, 11):
        print(f"    repo_{i}_*/")
        print(f"      *.py")
        print(f"      test_*.py")


if __name__ == "__main__":
    try:
        create_agent_benchmark()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
