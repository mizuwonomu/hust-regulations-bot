"""Package STOP-policy eval: capture, label review, replay và metric

Chỉ schema version được export ở đây để import package không kéo theo prompt,
client hay store; phần còn lại được gọi qua module sở hữu hoặc `cli`
"""

from stop_policy_eval.contracts import STOP_POLICY_SCHEMA_VERSION

__all__ = ["STOP_POLICY_SCHEMA_VERSION"]
