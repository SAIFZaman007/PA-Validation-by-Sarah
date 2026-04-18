from typing import Sequence, Union
 
from alembic import op
import sqlalchemy as sa
 
revision: str = '9792ad489538'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
 
def upgrade() -> None:
    pass
 
def downgrade() -> None:
    pass