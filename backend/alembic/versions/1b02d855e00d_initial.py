"""initial

Revision ID: 1b02d855e00d
Revises: 
Create Date: 2026-04-10 03:21:16.901228

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '1b02d855e00d'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. users (no dependencies)
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('public_id', sa.String(length=36), nullable=False),
    sa.Column('username', sa.String(length=80), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=True),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_public_id'), 'users', ['public_id'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # 2. conversations (depends on users)
    op.create_table('conversations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('public_id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('title', sa.String(length=255), nullable=True),
    sa.Column('active_mode', sa.Enum('NORMAL_CHAT', 'IMAGE_ITERATION', 'LIVE_INFO', 'FILE_ANALYSIS', name='conversationmode'), nullable=False),
    sa.Column('pending_followup_kind', sa.String(length=50), nullable=True),
    sa.Column('pending_followup_target', sa.Text(), nullable=True),
    sa.Column('allow_context_carryover', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_conversations_public_id'), 'conversations', ['public_id'], unique=True)
    op.create_index(op.f('ix_conversations_user_id'), 'conversations', ['user_id'], unique=False)

    # 3. messages (depends on conversations — turns FK added later)
    op.create_table('messages',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('public_id', sa.String(length=36), nullable=False),
    sa.Column('conversation_id', sa.Integer(), nullable=False),
    sa.Column('turn_id', sa.Integer(), nullable=True),
    sa.Column('role', sa.Enum('SYSTEM', 'USER', 'ASSISTANT', 'TOOL', name='messagerole'), nullable=False),
    sa.Column('content', sa.Text(), nullable=True),
    sa.Column('content_json', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_messages_conversation_id'), 'messages', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_messages_public_id'), 'messages', ['public_id'], unique=True)
    op.create_index(op.f('ix_messages_role'), 'messages', ['role'], unique=False)
    op.create_index(op.f('ix_messages_turn_id'), 'messages', ['turn_id'], unique=False)

    # 4. turns (depends on conversations and messages)
    op.create_table('turns',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('public_id', sa.String(length=36), nullable=False),
    sa.Column('conversation_id', sa.Integer(), nullable=False),
    sa.Column('sequence_number', sa.Integer(), nullable=False),
    sa.Column('user_message_id', sa.Integer(), nullable=True),
    sa.Column('assistant_message_id', sa.Integer(), nullable=True),
    sa.Column('planner_trace', sa.JSON(), nullable=True),
    sa.Column('final_plan', sa.JSON(), nullable=True),
    sa.Column('state_patch', sa.JSON(), nullable=True),
    sa.Column('status', sa.Enum('STARTED', 'COMPLETED', 'FAILED', 'CANCELLED', name='turnstatus'), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['assistant_message_id'], ['messages.id'], ),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),
    sa.ForeignKeyConstraint(['user_message_id'], ['messages.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('conversation_id', 'sequence_number', name='uq_turn_sequence_per_conversation')
    )
    op.create_index(op.f('ix_turns_conversation_id'), 'turns', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_turns_public_id'), 'turns', ['public_id'], unique=True)
    op.create_index(op.f('ix_turns_sequence_number'), 'turns', ['sequence_number'], unique=False)

    # 5. Add turn_id FK to messages now that turns table exists
    op.create_foreign_key(None, 'messages', 'turns', ['turn_id'], ['id'])

    # 6. session_tokens (depends on users)
    op.create_table('session_tokens',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('public_id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('access_token', sa.Text(), nullable=False),
    sa.Column('refresh_token', sa.Text(), nullable=False),
    sa.Column('is_revoked', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('access_token'),
    sa.UniqueConstraint('refresh_token')
    )
    op.create_index(op.f('ix_session_tokens_public_id'), 'session_tokens', ['public_id'], unique=True)
    op.create_index(op.f('ix_session_tokens_user_id'), 'session_tokens', ['user_id'], unique=False)

    # 7. memories (depends on users)
    op.create_table('memories',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('memory_type', sa.Enum('PROFILE', 'EPISODIC', 'WORKING', name='memorytype'), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('embedding', sa.Text(), nullable=True),
    sa.Column('salience_score', sa.Float(), nullable=False),
    sa.Column('metadata_json', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_memories_memory_type'), 'memories', ['memory_type'], unique=False)
    op.create_index(op.f('ix_memories_user_id'), 'memories', ['user_id'], unique=False)

    # 8. tool_events (depends on conversations and turns)
    op.create_table('tool_events',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('public_id', sa.String(length=36), nullable=False),
    sa.Column('conversation_id', sa.Integer(), nullable=False),
    sa.Column('turn_id', sa.Integer(), nullable=True),
    sa.Column('tool_name', sa.Enum('CHAT', 'WEB', 'IMAGE', 'FILE', 'MEMORY', name='toolname'), nullable=False),
    sa.Column('status', sa.Enum('PENDING', 'SUCCESS', 'FAILED', name='toolstatus'), nullable=False),
    sa.Column('input_text', sa.Text(), nullable=True),
    sa.Column('output_text', sa.Text(), nullable=True),
    sa.Column('payload_json', sa.JSON(), nullable=True),
    sa.Column('latency_ms', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),
    sa.ForeignKeyConstraint(['turn_id'], ['turns.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_tool_events_conversation_id'), 'tool_events', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_tool_events_public_id'), 'tool_events', ['public_id'], unique=True)
    op.create_index(op.f('ix_tool_events_status'), 'tool_events', ['status'], unique=False)
    op.create_index(op.f('ix_tool_events_tool_name'), 'tool_events', ['tool_name'], unique=False)
    op.create_index(op.f('ix_tool_events_turn_id'), 'tool_events', ['turn_id'], unique=False)

    # 9. artifacts (depends on conversations, turns, tool_events)
    op.create_table('artifacts',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('public_id', sa.String(length=36), nullable=False),
    sa.Column('conversation_id', sa.Integer(), nullable=False),
    sa.Column('turn_id', sa.Integer(), nullable=True),
    sa.Column('source_tool_event_id', sa.Integer(), nullable=True),
    sa.Column('parent_artifact_id', sa.Integer(), nullable=True),
    sa.Column('artifact_type', sa.Enum('IMAGE', 'FILE', 'WEB_RESULT', 'TEXT', name='artifacttype'), nullable=False),
    sa.Column('title', sa.String(length=255), nullable=True),
    sa.Column('storage_url', sa.Text(), nullable=True),
    sa.Column('inline_data', sa.Text(), nullable=True),
    sa.Column('prompt', sa.Text(), nullable=True),
    sa.Column('effective_prompt', sa.Text(), nullable=True),
    sa.Column('metadata_json', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),
    sa.ForeignKeyConstraint(['parent_artifact_id'], ['artifacts.id'], ),
    sa.ForeignKeyConstraint(['source_tool_event_id'], ['tool_events.id'], ),
    sa.ForeignKeyConstraint(['turn_id'], ['turns.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_artifacts_artifact_type'), 'artifacts', ['artifact_type'], unique=False)
    op.create_index(op.f('ix_artifacts_conversation_id'), 'artifacts', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_artifacts_public_id'), 'artifacts', ['public_id'], unique=True)
    op.create_index(op.f('ix_artifacts_turn_id'), 'artifacts', ['turn_id'], unique=False)

    # 10. uploads (depends on users, conversations, artifacts)
    op.create_table('uploads',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('public_id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('conversation_id', sa.Integer(), nullable=True),
    sa.Column('artifact_id', sa.Integer(), nullable=True),
    sa.Column('filename', sa.String(length=255), nullable=False),
    sa.Column('content_type', sa.String(length=255), nullable=True),
    sa.Column('storage_url', sa.Text(), nullable=True),
    sa.Column('size_bytes', sa.Integer(), nullable=True),
    sa.Column('metadata_json', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['artifact_id'], ['artifacts.id'], ),
    sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_uploads_conversation_id'), 'uploads', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_uploads_public_id'), 'uploads', ['public_id'], unique=True)
    op.create_index(op.f('ix_uploads_user_id'), 'uploads', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_table('uploads')
    op.drop_table('artifacts')
    op.drop_table('tool_events')
    op.drop_table('memories')
    op.drop_table('session_tokens')
    op.drop_table('turns')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('users')