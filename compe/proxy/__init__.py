"""ローカル層別プロキシ評価スイート（libero_proxy）。

本番採点には含まれない。摂動カテゴリ別の汎化性能をローカルで測るために、
compe/t1 と同じ CSV 登録方式で LIBERO-plus の変種タスクを登録する。
"""

from .register import register_proxy

__all__ = ["register_proxy"]
