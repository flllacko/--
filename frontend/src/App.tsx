import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent, Dispatch, FormEvent, SetStateAction } from 'react';

type NavItem = {
  path: string;
  label: string;
  icon: string;
};

type Page =
  | 'chat'
  | 'private'
  | 'public'
  | 'users'
  | 'logs'
  | 'login'
  | 'register'
  | 'profile'
  | 'not-found';

const navItems: NavItem[] = [
  { path: '/chat', label: '我的会话', icon: '▤' },
  { path: '/knowledge/private', label: '知识库管理', icon: '▱' },
  { path: '/profile', label: '个人中心', icon: '◎' },
];

const adminNavItems: NavItem[] = [
  { path: '/admin/users', label: '用户管理', icon: '♙' },
  { path: '/admin/logs', label: '操作记录', icon: '≡' },
];

const identities = [
  { key: 'student', label: '学生' },
  { key: 'parent', label: '家长' },
  { key: 'teacher', label: '老师' },
  { key: 'skip', label: '跳过' },
] as const;

type IdentityKey = (typeof identities)[number]['key'];

type SearchHit = {
  rank: number;
  document_id: number;
  chunk_index: number;
  title: string;
  module: string;
  text: string;
  locator: string;
  source_label: string;
  source_url: string;
  published_date: string;
  publisher_department: string;
  content_status: string;
  validity_status: string;
  is_private: boolean;
  score: number;
  match_types: string[];
};

type AnswerBody = {
  conclusion: string;
  conditions: string[];
  materials: string[];
  steps: string[];
  time: string;
  notes: string[];
  follow_up_questions: string[];
};

type AnswerResponse = {
  answer_status: 'completed' | 'no_reliable_evidence';
  answer: AnswerBody;
  sources: SearchHit[];
  retrieval: { hit_count: number };
  saved?: boolean;
  conversation_id?: number | null;
};

type User = {
  id: number;
  username: string;
  role: 'user' | 'admin';
  must_change_password: boolean;
  created_at: string;
  last_login_at: string | null;
};

type ConversationSummary = {
  id: number;
  title: string;
  identity: IdentityKey;
  is_pinned: boolean;
  created_at: string;
  updated_at: string;
  message_count: number;
  time_group: string;
};

type GuestEntry = {
  question: string;
  answer: AnswerBody;
  answer_status: AnswerResponse['answer_status'];
  sources: SearchHit[];
};

type ChatTurn = {
  draft?: string;
  draftAnswer?: AnswerBody;
  id: string;
  question: string;
  progress: string[];
  status: 'processing' | 'completed' | 'error';
  answer: AnswerResponse | null;
  error?: string;
};

type ConversationDetail = {
  id: number;
  title: string;
  identity: IdentityKey;
  is_pinned: boolean;
  messages: Array<{
    id: number;
    role: 'user' | 'assistant';
    content: string;
    status: string;
    answer?: AnswerBody & { answer_status?: AnswerResponse['answer_status'] };
    sources?: SearchHit[];
  }>;
};

type AdminUser = {
  id: number;
  username: string;
  created_at: string;
  last_login_at: string | null;
  status: string;
  must_change_password: boolean;
};

type AdminDocument = {
  id: number;
  stable_key: string;
  title: string;
  module: string;
  source_label: string;
  source_url: string | null;
  published_date: string | null;
  publisher_department: string | null;
  content_status: string;
  validity_status: string;
  processing_status: string;
  version: string | null;
  updated_at: string;
  chunk_count: number;
  error_message: string | null;
};

type OperationLog = {
  id: number;
  action: string;
  target_type: string;
  target_id: string | null;
  detail: Record<string, unknown>;
  created_at: string;
  actor_username: string | null;
};

type ProcessingTask = {
  id: number;
  document_id: number;
  title: string;
  status: string;
  error_message: string | null;
  created_at: string;
  finished_at: string | null;
};

type PrivateDocument = {
  id: number;
  title: string;
  filename: string;
  processing_status: '处理成功';
  created_at: string;
  updated_at: string;
};

type PrivateUploadResult = {
  filename: string;
  document_id?: number;
  task_id?: number;
  accepted: boolean;
  status: '处理成功' | '处理失败';
  error_message?: string | null;
};

const GUEST_SESSION_KEY = 'cdutetc_guest_session_v1';

async function apiJson<T>(input: RequestInfo | URL, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const response = await fetch(input, { ...init, headers, credentials: 'include' });
  const payload = (await response.json().catch(() => ({}))) as { detail?: string } & T;
  if (!response.ok) throw new Error(payload.detail || '请求失败，请稍后重试。');
  return payload;
}

function loadGuestEntries(): GuestEntry[] {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(GUEST_SESSION_KEY) || '[]') as unknown;
    return Array.isArray(parsed) ? parsed as GuestEntry[] : [];
  } catch {
    return [];
  }
}

const EXAMPLE_ROTATION_KEY = 'cdutetc_example_rotation_v1';
const ACTIVE_GUEST_KEY = 'cdutetc_active_guest_v1';

function readPartialString(raw: string, start: number): { text: string; end: number; closed: boolean } {
  let escaped = false;
  let end = start;
  for (; end < raw.length; end++) {
    if (!escaped && raw[end] === '"') break;
    if (!escaped && raw[end] === '\\') escaped = true;
    else escaped = false;
  }
  let part = raw.slice(start, end);
  // 网络包可能切在转义字符或 Unicode 转义中间，等待后续字符。
  for (let trim = 0; trim <= Math.min(6, part.length); trim++) {
    try { return { text: JSON.parse('"' + part.slice(0, part.length - trim) + '"') as string, end, closed: raw[end] === '"' }; } catch { /* 等待完整转义 */ }
  }
  return { text: '', end, closed: false };
}

function partialAnswer(raw: string): AnswerBody {
  const answer: AnswerBody = { conclusion: '', conditions: [], materials: [], steps: [], time: '', notes: [], follow_up_questions: [] };
  for (const field of ['conclusion', 'time'] as const) {
    const match = new RegExp('"' + field + '"\\s*:\\s*"').exec(raw);
    if (match) answer[field] = readPartialString(raw, match.index + match[0].length).text;
  }
  for (const field of ['conditions', 'materials', 'steps', 'notes', 'follow_up_questions'] as const) {
    const match = new RegExp('"' + field + '"\\s*:\\s*\\[').exec(raw);
    if (!match) continue;
    let pos = match.index + match[0].length;
    while (pos < raw.length) {
      while (/\s|,/.test(raw[pos] || '') && pos < raw.length) pos++;
      if (raw[pos] !== '"') break;
      const item = readPartialString(raw, pos + 1);
      if (item.text) answer[field].push(item.text);
      if (!item.closed) break;
      pos = item.end + 1;
    }
  }
  return answer;
}

function hasVisibleAnswer(answer: AnswerBody): boolean {
  return Boolean(
    answer.conclusion || answer.time || answer.conditions.length || answer.materials.length
    || answer.steps.length || answer.notes.length || answer.follow_up_questions.length,
  );
}

function loadActiveGuestTurns(): ChatTurn[] {
  try {
    const raw = sessionStorage.getItem(ACTIVE_GUEST_KEY);
    const entries = raw === null ? loadGuestEntries() : JSON.parse(raw);
    return Array.isArray(entries) ? entries.map(guestEntryToChatTurn) : [];
  } catch { return []; }
}

const examplePool = [
  '国家助学金有哪几个档次？',
  '休学、复学需要准备什么材料？',
  '新生宿舍和校园生活有哪些规定？',
  '介绍一下学校',
  '毕业生就业信息从哪里查看？',
  '转专业一般需要满足什么条件？',
  '学生医保报销需要准备什么？',
  '寝室使用明火有什么规定？',
  '专升本报名和缴费什么时候进行？',
];

let pageExampleSelection: string[] | null = null;

function getRotatedExamples(): string[] {
  try {
    if (pageExampleSelection) return pageExampleSelection;
    const previous = JSON.parse(sessionStorage.getItem(EXAMPLE_ROTATION_KEY) || '[]');
    const candidates = examplePool.filter(item => !Array.isArray(previous) || !previous.includes(item));
    for (let i = candidates.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
    }
    const selected = candidates.slice(0, 3);
    sessionStorage.setItem(EXAMPLE_ROTATION_KEY, JSON.stringify(selected));
    pageExampleSelection = selected;
    return pageExampleSelection;
  } catch {
    pageExampleSelection = pageExampleSelection ?? examplePool.slice(0, 3);
    return pageExampleSelection;
  }
}

function guestEntryToChatTurn(entry: GuestEntry, index: number): ChatTurn {
  return {
    id: `guest-${index}-${entry.question}`,
    question: entry.question,
    progress: ['已检索并整理回答'],
    status: 'completed',
    answer: {
      answer_status: entry.answer_status,
      answer: entry.answer,
      sources: entry.sources,
      retrieval: { hit_count: entry.sources.length },
      saved: false,
      conversation_id: null,
    },
  };
}

function readPath(): string {
  return window.location.pathname || '/chat';
}

function App() {
  const [path, setPath] = useState<string>(readPath);
  const [identity, setIdentity] = useState<IdentityKey>('student');
  const [question, setQuestion] = useState('');
  const [notice, setNotice] = useState('');
  const [answerResult, setAnswerResult] = useState<AnswerResponse | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [streamPreview, setStreamPreview] = useState('');
  const [user, setUser] = useState<User | null>(null);
  const [hasPrivateKnowledge, setHasPrivateKnowledge] = useState(false);
  const [usePrivateKnowledge, setUsePrivateKnowledge] = useState(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [guestEntries, setGuestEntries] = useState<GuestEntry[]>(loadGuestEntries);
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>(loadActiveGuestTurns);
  const [examplePrompts] = useState<string[]>(getRotatedExamples);
  const abortControllerRef = useRef<AbortController | null>(null);
  useEffect(() => {
    if (user) return;
    const entries = chatTurns.filter(turn => turn.status === 'completed' && turn.answer).map(turn => ({
      question: turn.question,
      answer: turn.answer!.answer,
      answer_status: turn.answer!.answer_status,
      sources: turn.answer!.sources,
    }));
    try { sessionStorage.setItem(ACTIVE_GUEST_KEY, JSON.stringify(entries)); } catch { /* 当前页面仍可继续聊天。 */ }
  }, [chatTurns, user]);

  useEffect(() => {
    const onPopState = () => setPath(readPath());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  useEffect(() => {
    void apiJson<{ authenticated: boolean; user: User | null }>('/api/auth/me')
      .then((payload) => setUser(payload.user))
      .catch(() => setUser(null));
  }, []);

  useEffect(() => {
    if (guestEntries.length) {
      sessionStorage.setItem(GUEST_SESSION_KEY, JSON.stringify(guestEntries));
    } else {
      sessionStorage.removeItem(GUEST_SESSION_KEY);
    }
  }, [guestEntries]);

  const loadConversations = async (currentUser: User): Promise<void> => {
    const payload = await apiJson<{ items: ConversationSummary[] }>('/api/conversations');
    setConversations(payload.items);
    if (!currentUser) setConversations([]);
  };

  useEffect(() => {
    if (!user) {
      setConversations([]);
      setActiveConversationId(null);
      return;
    }
    void loadConversations(user).catch((error: unknown) => {
      setNotice(error instanceof Error ? error.message : '历史会话加载失败。');
    });
  }, [user]);

  useEffect(() => {
    setUsePrivateKnowledge(false);
    if (!user || user.role === 'admin') {
      setHasPrivateKnowledge(false);
      return;
    }
    void apiJson<{ has_successful: boolean }>('/api/private/documents?limit=1')
      .then((payload) => setHasPrivateKnowledge(payload.has_successful))
      .catch(() => setHasPrivateKnowledge(false));
  }, [user]);

  const navigate = (nextPath: string): void => {
    if (nextPath === path) return;
    window.history.pushState({}, '', nextPath);
    setPath(nextPath);
    setNotice('');
  };

  const page = useMemo<Page>(() => {
    if (path === '/chat' || path === '/') return 'chat';
    if (path === '/knowledge/private') return 'private';
    if (path === '/knowledge/public') return 'public';
    if (path === '/admin/users') return 'users';
    if (path === '/admin/logs') return 'logs';
    if (path === '/login') return 'login';
    if (path === '/register') return 'register';
    if (path === '/profile') return 'profile';
    return 'not-found';
  }, [path]);

  const handleAuthenticated = async (nextUser: User): Promise<void> => {
    setUser(nextUser);
    if (guestEntries.length && window.confirm('发现本标签页的临时会话，是否转入账号历史？')) {
      try {
        await apiJson<{ conversation_id: number }>('/api/conversations/guest-import', {
          method: 'POST',
          body: JSON.stringify({ entries: guestEntries, identity }),
        });
        setGuestEntries([]);
        setNotice('临时会话已转入历史记录。');
      } catch (error) {
        setNotice(error instanceof Error ? error.message : '临时会话转入失败，仍保留在本标签页。');
      }
    }
    if (nextUser.must_change_password) {
      setNotice('管理员已重置你的密码，请先在个人中心修改密码。');
      navigate('/profile');
    } else {
      navigate('/chat');
    }
  };

  const handleLogout = async (): Promise<void> => {
    try {
      await apiJson<{ logged_out: boolean }>('/api/auth/logout', { method: 'POST' });
      setUser(null);
      setHasPrivateKnowledge(false);
      setUsePrivateKnowledge(false);
      setConversations([]);
      setActiveConversationId(null);
      setChatTurns([]);
      setAnswerResult(null);
      setNotice('已退出登录；访客仍可继续提问。');
      navigate('/chat');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '退出登录失败。');
    }
  };

  const handleAccountDeleted = (): void => {
    setUser(null);
    setHasPrivateKnowledge(false);
    setUsePrivateKnowledge(false);
    setConversations([]);
    setActiveConversationId(null);
    setChatTurns([]);
    setAnswerResult(null);
    setNotice('账号、历史会话和私人资料已删除。');
    navigate('/chat');
  };

  const selectConversation = async (conversationId: number): Promise<void> => {
    try {
      const detail = await apiJson<ConversationDetail>(`/api/conversations/${conversationId}`);
      const restoredTurns: ChatTurn[] = [];
      detail.messages.forEach((message) => {
        if (message.role === 'user') {
          restoredTurns.push({
            id: `history-${message.id}`,
            question: message.content,
            progress: ['已从历史会话加载'],
            status: 'completed',
            answer: null,
          });
          return;
        }
        const current = restoredTurns[restoredTurns.length - 1];
        if (current && message.answer) {
          const sources = message.sources || [];
          current.answer = {
            answer_status: message.answer.answer_status || 'completed',
            answer: message.answer,
            sources,
            retrieval: { hit_count: sources.length },
            saved: true,
            conversation_id: conversationId,
          };
        }
      });
      setChatTurns(restoredTurns);
      const latestAnswer = [...restoredTurns].reverse().find((turn) => turn.answer)?.answer || null;
      setAnswerResult(latestAnswer);
      setQuestion('');
      setIdentity(detail.identity);
      setUsePrivateKnowledge(false);
      setActiveConversationId(conversationId);
      setNotice(`已打开历史会话：${detail.title}`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '历史会话打开失败。');
    }
  };

  const startNewConversation = (): void => {
    if (isSearching) return;
    setActiveConversationId(null);
    setUsePrivateKnowledge(false);
    setQuestion('');
    setChatTurns([]);
    setAnswerResult(null);
    setStreamPreview('');
    setNotice('已开始新的空白会话。');
  };

  const renameConversation = async (conversationId: number): Promise<void> => {
    const current = conversations.find((item) => item.id === conversationId);
    const title = window.prompt('请输入新的会话标题', current?.title || '');
    if (title === null) return;
    try {
      await apiJson(`/api/conversations/${conversationId}`, {
        method: 'PATCH',
        body: JSON.stringify({ title }),
      });
      if (user) await loadConversations(user);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '会话重命名失败。');
    }
  };

  const toggleConversationPin = async (conversation: ConversationSummary): Promise<void> => {
    try {
      await apiJson(`/api/conversations/${conversation.id}`, {
        method: 'PATCH',
        body: JSON.stringify({ is_pinned: !conversation.is_pinned }),
      });
      if (user) await loadConversations(user);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '会话置顶操作失败。');
    }
  };

  const deleteConversation = async (conversationId: number): Promise<void> => {
    if (!window.confirm('删除后会同步删除该会话中的来源记录，确定继续吗？')) return;
    try {
      await apiJson(`/api/conversations/${conversationId}`, { method: 'DELETE' });
      if (activeConversationId === conversationId) startNewConversation();
      if (user) await loadConversations(user);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '会话删除失败。');
    }
  };

  const deleteConversations = async (conversationIds: number[]): Promise<void> => {
    if (!conversationIds.length || !window.confirm(`确定删除选中的 ${conversationIds.length} 个会话吗？`)) return;
    try {
      await apiJson('/api/conversations/batch-delete', {
        method: 'POST',
        body: JSON.stringify({ conversation_ids: conversationIds }),
      });
      if (activeConversationId && conversationIds.includes(activeConversationId)) startNewConversation();
      if (user) await loadConversations(user);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '批量删除失败。');
    }
  };

  const submitQuestion = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (!question.trim() || isSearching) return;
    const submittedQuestion = question.trim();
    const turnId = `turn-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setChatTurns((current) => [
      ...current,
      {
        id: turnId,
        question: submittedQuestion,
        progress: ['已收到问题'],
        status: 'processing',
        answer: null,
      },
    ]);
    setQuestion('');
    setIsSearching(true);
    setAnswerResult(null);
    setNotice('');
    try {
      const controller = new AbortController();
      abortControllerRef.current = controller;
      const response = await fetch('/api/answer/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          question: submittedQuestion,
          history: chatTurns.filter(turn => turn.status === 'completed' && turn.answer).slice(-4).flatMap(turn => [
            { role: 'user', content: turn.question.slice(0, 2000) },
            { role: 'assistant', content: JSON.stringify(turn.answer!.answer).slice(0, 2000) },
          ]),
          scope: 'current',
          limit: 6,
          identity,
          use_private_knowledge: usePrivateKnowledge,
          ...(activeConversationId ? { conversation_id: activeConversationId } : {}),
        }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const payload = (await response.json()) as { detail?: string };
        throw new Error(payload.detail || '检索失败');
      }
      if (!response.body) throw new Error('浏览器不支持流式响应');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let rawAnswer = '';
      const finalResult: { value: AnswerResponse | null } = { value: null };
      const handleEvent = (block: string): void => {
        const eventName = block.split('\n').find((line) => line.startsWith('event:'))?.slice(6).trim();
        const dataLine = block.split('\n').find((line) => line.startsWith('data:'))?.slice(5).trim();
        if (!eventName || !dataLine) return;
        const data = JSON.parse(dataLine) as { text?: string; message?: string; label?: string } & Partial<AnswerResponse>;
        if (eventName === 'progress' && data.label) {
          setChatTurns((current) => current.map((turn) => turn.id === turnId
            ? { ...turn, progress: turn.progress.includes(data.label as string) ? turn.progress : [...turn.progress, data.label as string] }
            : turn));
        } else if (eventName === 'answer_delta' && data.text) {
          setChatTurns(current => current.map(turn => turn.id === turnId
            ? { ...turn, draft: (turn.draft || '') + data.text } : turn));
        } else if (eventName === 'delta' && data.text) {
          rawAnswer += data.text;
          const draftAnswer = partialAnswer(rawAnswer);
          if (hasVisibleAnswer(draftAnswer)) {
            setChatTurns(current => current.map(turn => turn.id === turnId ? { ...turn, draft: draftAnswer.conclusion, draftAnswer } : turn));
          }
          setStreamPreview((current) => current + data.text);
        } else if (eventName === 'final') {
          finalResult.value = data as AnswerResponse;
        } else if (eventName === 'error') {
          throw new Error(data.message || '模型生成失败');
        }
      };
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() || '';
        blocks.forEach(handleEvent);
        if (done) break;
      }
      if (buffer.trim()) handleEvent(buffer);
      if (!finalResult.value) throw new Error('回答连接已结束，但未收到完整回答，请重新发送问题。');
      if (finalResult.value) {
        const result = finalResult.value;
        setAnswerResult(result);
        setStreamPreview('');
        setChatTurns((current) => current.map((turn) => turn.id === turnId
          ? { ...turn, status: 'completed', answer: result, progress: [...turn.progress, '回答已完成'] }
          : turn));
        if (!user) {
          setGuestEntries((current) => [
            ...current,
            {
              question: submittedQuestion,
              answer: result.answer,
              answer_status: result.answer_status,
              sources: result.sources,
            },
          ]);
        } else if (result.conversation_id) {
          setActiveConversationId(result.conversation_id);
          void loadConversations(user);
        }
        setNotice('');
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        setStreamPreview('');
        setChatTurns((current) => current.map((turn) => turn.id === turnId
          ? { ...turn, status: 'error', error: '已停止生成；本次未保存未完成回答。', progress: [...turn.progress, '回答已停止'] }
          : turn));
        setNotice('已停止生成；本次未保存未完成回答。');
      } else {
        setStreamPreview('');
        const errorMessage = error instanceof Error ? error.message : '检索失败，请确认后端和 Ollama 已启动。';
        setChatTurns((current) => current.map((turn) => turn.id === turnId
          ? { ...turn, status: 'error', error: errorMessage, progress: [...turn.progress, '回答未完成'] }
          : turn));
        setNotice(errorMessage);
      }
    } finally {
      abortControllerRef.current = null;
      setIsSearching(false);
    }
  };

  const stopAnswer = (): void => {
    abortControllerRef.current?.abort();
  };

  const selectExample = (example: string): void => {
    setQuestion(example);
    setAnswerResult(null);
    setStreamPreview('');
    setNotice('问题已放入输入框，提交后会先检索官方资料。');
  };

  const selectFollowUp = (followUp: string): void => {
    setQuestion(followUp);
    setAnswerResult(null);
    setStreamPreview('');
    setNotice('已将补充问题放入输入框，请确认后发送。');
  };

  return (
    <div className="app-shell">
      <Sidebar path={path} navigate={navigate} user={user} />
      <main className="main-area">
        {page === 'chat' && (
          <ChatPage
            navigate={navigate}
            identity={identity}
            setIdentity={setIdentity}
            question={question}
            setQuestion={setQuestion}
            submitQuestion={submitQuestion}
            selectExample={selectExample}
            examplePrompts={examplePrompts}
            notice={notice}
            answerResult={answerResult}
            chatTurns={chatTurns}
            isSearching={isSearching}
            streamPreview={streamPreview}
            stopAnswer={stopAnswer}
            selectFollowUp={selectFollowUp}
            user={user}
            hasPrivateKnowledge={hasPrivateKnowledge}
            usePrivateKnowledge={usePrivateKnowledge}
            setUsePrivateKnowledge={setUsePrivateKnowledge}
            conversations={conversations}
            guestEntries={guestEntries}
            activeConversationId={activeConversationId}
            onNewConversation={startNewConversation}
            onSelectConversation={(id) => void selectConversation(id)}
            onRenameConversation={(id) => void renameConversation(id)}
            onTogglePin={(conversation) => void toggleConversationPin(conversation)}
            onDeleteConversation={(id) => void deleteConversation(id)}
            onBatchDelete={(ids) => void deleteConversations(ids)}
          />
        )}
        {page === 'private' && (user?.role === 'admin'
          ? <PublicDocumentsPage />
          : <PrivateKnowledgePage onBack={() => navigate('/chat')} onAvailabilityChange={setHasPrivateKnowledge} />)}
        {page === 'public' && (user?.role === 'admin'
          ? <PublicDocumentsPage />
          : <AccessDeniedPage onBack={() => navigate('/chat')} />)}
        {page === 'users' && (user?.role === 'admin'
          ? <AdminUsersPage />
          : <AccessDeniedPage onBack={() => navigate('/chat')} />)}
        {page === 'logs' && (user?.role === 'admin'
          ? <AdminLogsPage />
          : <AccessDeniedPage onBack={() => navigate('/chat')} />)}
        {page === 'profile' && (
          <ProfilePage
            user={user}
            navigate={navigate}
            onLogout={() => void handleLogout()}
            onDeleted={handleAccountDeleted}
          />
        )}
        {page === 'login' && <AuthPage mode="login" navigate={navigate} onAuthenticated={(nextUser) => void handleAuthenticated(nextUser)} />}
        {page === 'register' && <AuthPage mode="register" navigate={navigate} onAuthenticated={(nextUser) => void handleAuthenticated(nextUser)} />}
        {page === 'not-found' && (
          <PlaceholderPage
            title="页面不存在"
            eyebrow="404"
            description="返回聊天主界面继续使用。"
            actionLabel="返回聊天"
            onAction={() => navigate('/chat')}
          />
        )}
      </main>
    </div>
  );
}

type SidebarProps = {
  path: string;
  navigate: (nextPath: string) => void;
  user: User | null;
};

function Sidebar({ path, navigate, user }: SidebarProps) {
  const activePath = path === '/' ? '/chat' : path;

  const goTo = (nextPath: string): void => {
    if (nextPath !== '/chat' && !user) {
      window.alert('“知识库管理”和“个人中心”需要登录后使用。');
      navigate('/login');
      return;
    }
    if (nextPath === '/knowledge/private' && user?.role === 'admin') {
      navigate('/knowledge/public');
      return;
    }
    navigate(nextPath);
  };

  return (
    <aside className="sidebar">
      <div className="brand-block">
        <div className="brand-mark">问</div>
        <div>
          <div className="brand-name">学生服务</div>
          <div className="brand-subtitle">问答助手</div>
        </div>
      </div>

      <div className="sidebar-section-label">工作区</div>
      <nav className="sidebar-nav" aria-label="主导航">
        {navItems.map((item) => (
          <NavButton
            key={item.path}
            item={item}
            active={activePath === item.path}
            onClick={() => goTo(item.path)}
          />
        ))}
      </nav>

      {user?.role === 'admin' && (
        <>
          <div className="sidebar-section-label admin-label">管理后台</div>
          <nav className="sidebar-nav" aria-label="管理后台导航">
            {adminNavItems.map((item) => (
              <NavButton
                key={item.path}
                item={item}
                active={activePath === item.path}
                onClick={() => goTo(item.path)}
              />
            ))}
          </nav>
        </>
      )}

      <div className="sidebar-account">
        {user ? (
          <button className="account-chip" type="button" onClick={() => navigate('/profile')}>
            <span className="account-avatar">{user.username.slice(0, 1)}</span>
            <span><strong>{user.username}</strong><small>{user.role === 'admin' ? '管理员' : '普通用户'}</small></span>
          </button>
        ) : (
          <div className="guest-account">
            <span>当前为访客</span>
            <div><button type="button" onClick={() => navigate('/login')}>登录</button><button type="button" onClick={() => navigate('/register')}>注册</button></div>
          </div>
        )}
      </div>

      <div className="sidebar-bottom">
        <div className="local-status">
          <span className="status-dot" />
          本地资料模式
        </div>
        <div className="version-text">本地问答正式版 · v1.0</div>
      </div>
    </aside>
  );
}

type NavButtonProps = {
  item: NavItem;
  active: boolean;
  onClick: () => void;
};

function NavButton({ item, active, onClick }: NavButtonProps) {
  const className = 'nav-button' + (active ? ' active' : '');

  return (
    <button className={className} type="button" onClick={onClick}>
      <span className="nav-icon" aria-hidden="true">{item.icon}</span>
      <span>{item.label}</span>
    </button>
  );
}

type TopBarProps = {
  identity: IdentityKey;
  setIdentity: Dispatch<SetStateAction<IdentityKey>>;
  user: User | null;
  navigate: (nextPath: string) => void;
};

function TopBar({ identity, setIdentity, user, navigate }: TopBarProps) {
  const handleIdentityChange = (event: ChangeEvent<HTMLSelectElement>): void => {
    setIdentity(event.target.value as IdentityKey);
  };

  return (
    <header className="topbar">
      <div>
        <div className="topbar-kicker">成都理工大学工程技术学院</div>
        <h1>学生服务问答助手</h1>
      </div>
      <div className="topbar-actions">
        <span className="local-badge"><span className="status-dot" />本地优先</span>
        <label className="identity-picker">
          <span className="identity-label">回答身份</span>
          <select value={identity} onChange={handleIdentityChange} aria-label="选择回答身份">
            {identities.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
          </select>
        </label>
        <button className="avatar-button" type="button" aria-label="打开个人中心" onClick={() => navigate(user ? '/profile' : '/login')}>
          {user ? user.username.slice(0, 1) : '访'}
        </button>
      </div>
    </header>
  );
}

type ChatPageProps = {
  navigate: (nextPath: string) => void;
  identity: IdentityKey;
  setIdentity: Dispatch<SetStateAction<IdentityKey>>;
  question: string;
  setQuestion: Dispatch<SetStateAction<string>>;
  submitQuestion: (event: FormEvent<HTMLFormElement>) => void;
  selectExample: (example: string) => void;
  examplePrompts: string[];
  notice: string;
  answerResult: AnswerResponse | null;
  chatTurns: ChatTurn[];
  isSearching: boolean;
  streamPreview: string;
  stopAnswer: () => void;
  selectFollowUp: (question: string) => void;
  user: User | null;
  hasPrivateKnowledge: boolean;
  usePrivateKnowledge: boolean;
  setUsePrivateKnowledge: Dispatch<SetStateAction<boolean>>;
  conversations: ConversationSummary[];
  guestEntries: GuestEntry[];
  activeConversationId: number | null;
  onNewConversation: () => void;
  onSelectConversation: (conversationId: number) => void;
  onRenameConversation: (conversationId: number) => void;
  onTogglePin: (conversation: ConversationSummary) => void;
  onDeleteConversation: (conversationId: number) => void;
  onBatchDelete: (conversationIds: number[]) => void;
};

function ChatPage({
  navigate,
  identity,
  setIdentity,
  question,
  setQuestion,
  submitQuestion,
  selectExample,
  examplePrompts,
  notice,
  answerResult,
  chatTurns,
  isSearching,
  streamPreview,
  stopAnswer,
  selectFollowUp,
  user,
  hasPrivateKnowledge,
  usePrivateKnowledge,
  setUsePrivateKnowledge,
  conversations,
  guestEntries,
  activeConversationId,
  onNewConversation,
  onSelectConversation,
  onRenameConversation,
  onTogglePin,
  onDeleteConversation,
  onBatchDelete,
}: ChatPageProps) {
  const selectedIdentity = identities.find((item) => item.key === identity);
  const [historyOpen, setHistoryOpen] = useState(false);
  const scrollArea = useRef<HTMLDivElement>(null);
  const followLatest = useRef(true);
  const previousTurnCount = useRef(0);
  useEffect(() => {
    const area = scrollArea.current;
    if (area && (followLatest.current || chatTurns.length !== previousTurnCount.current)) {
      area.scrollTop = area.scrollHeight;
    }
    previousTurnCount.current = chatTurns.length;
  }, [chatTurns]);
  const [suggestions, setSuggestions] = useState(examplePrompts);
  const refreshSuggestions = () => {
    pageExampleSelection = null;
    setSuggestions(getRotatedExamples());
  };
  const newChat = () => {
    onNewConversation();
    refreshSuggestions();
    setHistoryOpen(false);
  };

  return (
    <div className="page chat-page">
      <TopBar identity={identity} setIdentity={setIdentity} user={user} navigate={navigate} />
      <div className="chat-toolbar">
        <button type="button" disabled={isSearching} onClick={newChat}>＋ 新建会话</button>
        <button type="button" disabled={isSearching} aria-expanded={historyOpen} onClick={() => setHistoryOpen(!historyOpen)}>历史会话 {historyOpen ? '⌃' : '⌄'}</button>
      </div>
      <div className="chat-workspace">
        {historyOpen && <div className="history-popover">
        <HistoryPanel
          user={user}
          conversations={conversations}
          guestEntries={guestEntries}
          activeConversationId={activeConversationId}
          onNewConversation={newChat}
          onSelectConversation={(id) => { onSelectConversation(id); setHistoryOpen(false); }}
          onRenameConversation={onRenameConversation}
          onTogglePin={onTogglePin}
          onDeleteConversation={onDeleteConversation}
          onBatchDelete={onBatchDelete}
        />
        </div>}
        <section className="chat-content">
        <div className="chat-scroll-area" ref={scrollArea} onScroll={(event) => {
          const area = event.currentTarget;
          followLatest.current = area.scrollHeight - area.scrollTop - area.clientHeight < 100;
        }}>
        {chatTurns.length === 0 ? (
          <>
            <div className="welcome-orbit" aria-hidden="true"><span>问</span></div>
            <div className="welcome-copy">
              <div className="eyebrow">欢迎使用</div>
              <h2>你想了解学校的什么信息？</h2>
              <p>我会先理解问题，再检索官方资料、核对证据并整理回答。</p>
            </div>

            <div className="example-grid" aria-label="问题示例">
              {suggestions.map((example) => (
                <button className="example-card" type="button" key={example} onClick={() => selectExample(example)}>
                  <span className="example-arrow">↗</span>
                  <span>{example}</span>
                </button>
              ))}
            </div>
            <button type="button" className="refresh-suggestions" onClick={refreshSuggestions}>↻ 换一换</button>
          </>
        ) : (
          <div className="chat-thread" aria-live="polite">
            {chatTurns.map((turn) => (
              <ChatTurnView key={turn.id} turn={turn} onFollowUp={selectFollowUp} />
            ))}
          </div>
        )}
        </div>

        <div className="composer-dock">
        <form className="composer" onSubmit={submitQuestion}>
          <textarea
            value={question}
            onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setQuestion(event.target.value)}
            placeholder="输入你想了解的问题，例如：休学需要哪些材料？"
            rows={2}
            aria-label="输入问题"
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                if (!isSearching && question.trim()) event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <div className="composer-footer">
            <div className="composer-options">
              <span className="composer-hint">回答身份：{selectedIdentity?.label}</span>
              {user?.role === 'user' && hasPrivateKnowledge && (
                <label className="private-toggle">
                  <input
                    type="checkbox"
                    checked={usePrivateKnowledge}
                    onChange={(event) => setUsePrivateKnowledge(event.target.checked)}
                  />
                  添加私人知识库
                </label>
              )}
            </div>
            {isSearching ? (
              <button className="stop-button" type="button" onClick={stopAnswer}>停止</button>
            ) : (
              <button className="send-button" type="submit" aria-label="提交问题" disabled={!question.trim()}>↑</button>
            )}
          </div>
        </form>

        {notice && (chatTurns.length === 0 || notice.includes('失败') || notice.includes('停止')) && <div className="honest-notice" role="status">{notice}</div>}
        <p className="privacy-note">Enter 发送 · Shift + Enter 换行 · 具体办理要求请核对引用的官方资料</p>
        </div>
        </section>
      </div>
    </div>
  );
}

function ChatTurnView({ turn, onFollowUp }: { turn: ChatTurn; onFollowUp: (question: string) => void }) {
  return (
    <article className="chat-turn">
      <div className="message-bubble user-message">
        <div className="message-role">你</div>
        <p>{turn.question}</p>
      </div>
      <div className="assistant-message">
        <div className="assistant-heading">
          <span className="assistant-avatar">问</span>
          <div><strong>学生服务问答助手</strong><small>本地资料优先</small></div>
        </div>
        {turn.status === 'processing' && !turn.draft && !turn.draftAnswer && <p className="reply-status" role="status">正在准备回答…</p>}
        {turn.error && <div className="chat-error" role="alert">{turn.error}</div>}
        {(turn.draft || turn.draftAnswer || turn.answer) && <AnswerPanel result={turn.answer} draft={turn.draft} draftAnswer={turn.draftAnswer} onFollowUp={onFollowUp} />}
        {turn.answer && <EvidenceDisclosure sources={turn.answer.sources} />}
      </div>
    </article>
  );
}

const HIDDEN_EVIDENCE_SECTIONS = [
  '资料信息', '可回答的问题', '能支撑的学生自然问法', '可回答的自然问法',
  '可回答范围', '使用边界', '整理边界', '时效边界', '证据边界',
  '资料用途', '已隔离或处理失败', '可作为问答候选',
];

function evidenceDisplayText(rawText: string): string {
  const lines = rawText.replace(/\r\n/g, '\n').split('\n');
  const visible: string[] = [];
  let hiddenSection = false;
  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const section = /^##\s+(.+?)\s*$/.exec(line)?.[1];
    if (section) {
      hiddenSection = HIDDEN_EVIDENCE_SECTIONS.some((name) => section === name || section.startsWith(name));
      if (!hiddenSection && !/正文|原文|核心条款|要点|摘录/.test(section)) visible.push(section);
      continue;
    }
    if (hiddenSection || /^#\s+/.test(line)) continue;
    if (/^[-*]\s*(资料编号|处理状态|内容状态|所属模块|资料类型|资料级别|来源文字|发布日期|时效状态|有效性初判|抓取日期|原文链接|检索别名|正文指纹|整理备注)[:：]/.test(line)) continue;
    visible.push(line);
  }
  return visible
    .join('\n')
    .replace(/\[([^\]]+)]\([^\)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^[-*]\s+/gm, '• ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function EvidenceDisclosure({ sources }: { sources: SearchHit[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;
  return (
    <details
      className="evidence-panel evidence-disclosure"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="evidence-summary">
        <span><span className="eyebrow">参考资料</span><strong>{sources.length} 条引用片段</strong></span>
        <span className="evidence-summary-action">{open ? '收起原文片段⌃' : '展开原文片段⌄'}</span>
      </summary>
      <div className="evidence-list">
        {sources.map((hit) => (
          <article className="evidence-card" key={`${hit.document_id}-${hit.chunk_index}`}>
            <h4>{hit.title}</h4>
            <p>{evidenceDisplayText(hit.text) || '该片段没有可直接展示的正文，请通过原文链接核对。'}</p>
            <div className="evidence-source">
              <span>{hit.source_label || '官方来源'}</span>
              {hit.published_date && <span>{hit.published_date}</span>}
              {hit.source_url && <a href={hit.source_url} target="_blank" rel="noreferrer">查看原文 ↗</a>}
            </div>
          </article>
        ))}
      </div>
    </details>
  );
}

type HistoryPanelProps = {
  user: User | null;
  conversations: ConversationSummary[];
  guestEntries: GuestEntry[];
  activeConversationId: number | null;
  onNewConversation: () => void;
  onSelectConversation: (conversationId: number) => void;
  onRenameConversation: (conversationId: number) => void;
  onTogglePin: (conversation: ConversationSummary) => void;
  onDeleteConversation: (conversationId: number) => void;
  onBatchDelete: (conversationIds: number[]) => void;
};

function HistoryPanel({
  user,
  conversations,
  guestEntries,
  activeConversationId,
  onNewConversation,
  onSelectConversation,
  onRenameConversation,
  onTogglePin,
  onDeleteConversation,
  onBatchDelete,
}: HistoryPanelProps) {
  const [query, setQuery] = useState('');
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const filtered = conversations.filter((item) => item.title.toLowerCase().includes(query.trim().toLowerCase()));
  const groups = ['今天', '昨天', '7天内', '30天内', '更早'];

  const toggleSelected = (id: number): void => {
    setSelectedIds((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current, id]);
  };

  return (
    <aside className="history-panel" aria-label="历史会话">
      <div className="history-heading">
        <div><div className="eyebrow">会话</div><h2>{user ? '我的会话' : '临时会话'}</h2></div>
        <button className="new-chat-button" type="button" onClick={onNewConversation} aria-label="新建会话">＋</button>
      </div>
      {user ? (
        <>
          <input className="history-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索会话标题" aria-label="搜索会话标题" />
          {selectedIds.length > 0 && <button className="batch-delete-button" type="button" onClick={() => { onBatchDelete(selectedIds); setSelectedIds([]); }}>删除选中（{selectedIds.length}）</button>}
          <div className="history-list">
            {groups.map((group) => {
              const items = filtered.filter((item) => item.time_group === group);
              if (!items.length) return null;
              return (
                <div className="history-group" key={group}>
                  <div className="history-group-label">{group}</div>
                  {items.map((conversation) => (
                    <div className={'history-item' + (activeConversationId === conversation.id ? ' active' : '')} key={conversation.id}>
                      <input type="checkbox" checked={selectedIds.includes(conversation.id)} onChange={() => toggleSelected(conversation.id)} aria-label={`选择${conversation.title}`} />
                      <button className="history-item-title" type="button" onClick={() => onSelectConversation(conversation.id)}>{conversation.is_pinned && '📌 '}{conversation.title}</button>
                      <button className="history-item-action" type="button" onClick={() => onTogglePin(conversation)} aria-label={conversation.is_pinned ? '取消置顶' : '置顶'}>{conversation.is_pinned ? '★' : '☆'}</button>
                      <button className="history-item-action" type="button" onClick={() => onRenameConversation(conversation.id)} aria-label="重命名">⋯</button>
                      <button className="history-item-action danger" type="button" onClick={() => onDeleteConversation(conversation.id)} aria-label="删除">×</button>
                    </div>
                  ))}
                </div>
              );
            })}
            {!filtered.length && <div className="history-empty">还没有匹配的历史会话。</div>}
          </div>
        </>
      ) : (
        <div className="guest-history-copy">
          <p>访客提问只保存在当前标签页，刷新后仍可继续，关闭标签页后清除。</p>
          <strong>{guestEntries.length ? `已完成 ${guestEntries.length} 个问题` : '当前还没有临时问答'}</strong>
          {guestEntries.length > 0 && <button className="secondary-button full-width" type="button" onClick={() => window.alert('登录或注册后可将这些临时问答转入历史。')}>转入历史</button>}
          <button className="text-button" type="button" onClick={() => window.alert('登录后可以保存完整回答、来源片段和历史会话。')}>为什么要登录？</button>
        </div>
      )}
    </aside>
  );
}

function AnswerPanel({ result, draft, draftAnswer, onFollowUp }: { result: AnswerResponse | null; draft?: string; draftAnswer?: AnswerBody; onFollowUp: (question: string) => void }) {
  const answer = result?.answer ?? draftAnswer;
  return (
    <section className="answer-panel" aria-label={result ? '回答内容' : '正在生成回答'} aria-busy={!result}>
      <p className="answer-conclusion" style={{ whiteSpace: 'pre-wrap' }}>{answer?.conclusion ?? draft}</p>
      {answer && <>
      <AnswerList label="适用条件" items={answer.conditions} />
      <AnswerList label="所需材料" items={answer.materials} />
      <AnswerList label="办理流程" items={answer.steps} ordered />
      {answer.time && <div className="answer-row"><strong>时间</strong><span>{answer.time}</span></div>}
      <AnswerList label="注意事项" items={answer.notes} />
      {answer.follow_up_questions.length > 0 && (
        <div className="follow-up-panel">
          <strong>需要补充</strong>
          <div>{answer.follow_up_questions.map((question) => (
            <button type="button" key={question} disabled={!result} onClick={() => onFollowUp(question)}>{question}</button>
          ))}</div>
        </div>
      )}
      </>}
    </section>
  );
}

function AnswerList({ label, items, ordered = false }: { label: string; items: string[]; ordered?: boolean }) {
  if (!items.length) return null;
  const ListTag = ordered ? 'ol' : 'ul';
  return (
    <div className="answer-row">
      <strong>{label}</strong>
      <ListTag>{items.map((item) => <li key={item}>{item}</li>)}</ListTag>
    </div>
  );
}

function AccessDeniedPage({ onBack }: { onBack: () => void }) {
  return <PlaceholderPage title="没有权限" eyebrow="管理后台" description="只有管理员可以访问此页面。" actionLabel="返回聊天" onAction={onBack} />;
}

function AdminUsersPage() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [notice, setNotice] = useState('正在加载普通用户……');

  const loadUsers = async (): Promise<void> => {
    try {
      const payload = await apiJson<{ items: AdminUser[] }>('/api/admin/users');
      setUsers(payload.items);
      setNotice(payload.items.length ? '' : '目前还没有普通用户。');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '用户列表加载失败。');
    }
  };

  useEffect(() => { void loadUsers(); }, []);

  const resetPassword = async (user: AdminUser): Promise<void> => {
    if (!window.confirm(`确认把 ${user.username} 的密码重置为 123456 吗？`)) return;
    try {
      await apiJson(`/api/admin/users/${user.id}/reset-password`, { method: 'POST' });
      setNotice(`已重置 ${user.username}；该用户下次登录必须修改密码。`);
      await loadUsers();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '密码重置失败。');
    }
  };

  return (
    <div className="page inner-page">
      <div className="inner-header"><div><div className="eyebrow">管理后台</div><h2>用户管理</h2></div><span className="admin-safety-note">不显示密码、问答正文或私人资料</span></div>
      {notice && <div className="honest-notice" role="status">{notice}</div>}
      <div className="admin-table-wrap"><table className="admin-table"><thead><tr><th>用户名</th><th>注册时间</th><th>最近登录</th><th>状态</th><th>操作</th></tr></thead><tbody>{users.map((user) => <tr key={user.id}><td>{user.username}</td><td>{new Date(user.created_at).toLocaleString()}</td><td>{user.last_login_at ? new Date(user.last_login_at).toLocaleString() : '未登录'}</td><td>{user.status}{user.must_change_password && <small className="must-change-tag">待改密</small>}</td><td><button className="table-action" type="button" onClick={() => void resetPassword(user)}>重置密码</button></td></tr>)}</tbody></table></div>
    </div>
  );
}

function PublicDocumentsPage() {
  const [documents, setDocuments] = useState<AdminDocument[]>([]);
  const [tasks, setTasks] = useState<ProcessingTask[]>([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState('');
  const [validity, setValidity] = useState('');
  const [notice, setNotice] = useState('正在加载公共资料……');
  const [preview, setPreview] = useState<{ title: string; source_text: string } | null>(null);
  const [uploadModule, setUploadModule] = useState('学生事务');
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);

  const loadDocuments = async (): Promise<void> => {
    try {
      const params = new URLSearchParams({ q: query, validity_status: validity, limit: '100' });
      const payload = await apiJson<{ items: AdminDocument[]; total: number }>(`/api/admin/documents?${params.toString()}`);
      setDocuments(payload.items);
      setTotal(payload.total);
      setNotice('');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '公共资料加载失败。');
    }
  };

  const loadProcessing = async (): Promise<void> => {
    try {
      const payload = await apiJson<{ items: ProcessingTask[] }>('/api/admin/processing');
      setTasks(payload.items);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '处理队列状态加载失败。');
    }
  };

  useEffect(() => { void loadDocuments(); void loadProcessing(); }, [validity]);

  const changeValidity = async (document: AdminDocument): Promise<void> => {
    const action = document.validity_status === '有效' ? '停用' : '恢复';
    if (!window.confirm(`确认${action}《${document.title}》吗？`)) return;
    try {
      await apiJson(`/api/admin/documents/${document.id}/${action === '停用' ? 'disable' : 'restore'}`, { method: 'POST' });
      await loadDocuments();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : `${action}资料失败。`);
    }
  };

  const batchChangeValidity = async (nextStatus: '有效' | '无效'): Promise<void> => {
    if (!selectedIds.length) {
      setNotice('请先勾选资料。');
      return;
    }
    const action = nextStatus === '有效' ? '恢复' : '停用';
    if (!window.confirm(`确认${action}选中的 ${selectedIds.length} 条公共资料吗？`)) return;
    try {
      await apiJson('/api/admin/documents/batch-validity', {
        method: 'POST',
        body: JSON.stringify({ document_ids: selectedIds, validity_status: nextStatus }),
      });
      setSelectedIds([]);
      setNotice(`已${action} ${selectedIds.length} 条公共资料。`);
      await loadDocuments();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : `批量${action}资料失败。`);
    }
  };

  const retryDocument = async (document: AdminDocument): Promise<void> => {
    try {
      await apiJson(`/api/admin/documents/${document.id}/retry`, { method: 'POST' });
      setNotice(`已重新加入《${document.title}》的处理队列。`);
      await loadDocuments();
      await loadProcessing();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '资料重试失败。');
    }
  };

  const cancelTask = async (task: ProcessingTask): Promise<void> => {
    if (!window.confirm(`确认取消《${task.title}》的排队任务吗？`)) return;
    try {
      await apiJson(`/api/admin/processing/${task.id}/cancel`, { method: 'POST' });
      await loadDocuments();
      await loadProcessing();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '取消处理任务失败。');
    }
  };

  const deleteDocument = async (document: AdminDocument): Promise<void> => {
    if (!window.confirm(`确认准备删除《${document.title}》吗？`)) return;
    try {
      const plan = await apiJson<{ title: string; chunk_count: number; source_record_count: number; task_count: number; confirmation_token: string }>(`/api/admin/documents/${document.id}/delete-plan`, { method: 'POST' });
      if (!window.confirm(`将永久删除《${plan.title}》及 ${plan.chunk_count} 个切片、${plan.source_record_count} 条来源记录和 ${plan.task_count} 条任务记录，确定继续吗？`)) return;
      await apiJson(`/api/admin/documents/delete-confirm`, { method: 'POST', body: JSON.stringify({ confirmation_token: plan.confirmation_token }) });
      setNotice(`已删除《${document.title}》。`);
      await loadDocuments();
      await loadProcessing();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '资料删除失败。');
    }
  };

  const batchDeleteDocuments = async (): Promise<void> => {
    if (!selectedIds.length) {
      setNotice('请先勾选资料。');
      return;
    }
    if (!window.confirm(`确认准备删除选中的 ${selectedIds.length} 条公共资料吗？`)) return;
    try {
      const plan = await apiJson<{ plans: Array<{ title: string; chunk_count: number; source_record_count: number; task_count: number; confirmation_token: string }> }>('/api/admin/documents/batch-delete-plan', {
        method: 'POST',
        body: JSON.stringify({ document_ids: selectedIds }),
      });
      const chunks = plan.plans.reduce((sum, item) => sum + item.chunk_count, 0);
      const sources = plan.plans.reduce((sum, item) => sum + item.source_record_count, 0);
      const tasks = plan.plans.reduce((sum, item) => sum + item.task_count, 0);
      if (!window.confirm(`将永久删除 ${plan.plans.length} 条资料、${chunks} 个切片、${sources} 条来源记录和 ${tasks} 条任务记录，确定继续吗？`)) return;
      const result = await apiJson<{ deleted: unknown[]; failed: Array<{ message: string }> }>('/api/admin/documents/batch-delete-confirm', {
        method: 'POST',
        body: JSON.stringify({ confirmation_tokens: plan.plans.map((item) => item.confirmation_token) }),
      });
      setSelectedIds([]);
      setNotice(`已删除 ${result.deleted.length} 条资料${result.failed.length ? `，${result.failed.length} 条失败：${result.failed.map((item) => item.message).join('；')}` : '。'}`);
      await loadDocuments();
      await loadProcessing();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '批量删除资料失败。');
    }
  };

  const uploadDocuments = async (): Promise<void> => {
    if (!uploadFiles?.length) {
      setNotice('请先选择要上传的公共资料。');
      return;
    }
    const formData = new FormData();
    formData.append('module', uploadModule);
    Array.from(uploadFiles).forEach((file) => formData.append('file', file));
    try {
      const payload = await apiJson<{ items: Array<{ accepted: boolean; filename: string; message?: string }> }>('/api/admin/documents/upload', { method: 'POST', body: formData });
      const accepted = payload.items.filter((item) => item.accepted).length;
      const failed = payload.items.filter((item) => !item.accepted);
      setNotice(`已接收 ${accepted} 个文件进入队列${failed.length ? `，${failed.length} 个文件失败：${failed.map((item) => `${item.filename}（${item.message || '未知原因'}）`).join('；')}` : '。'}`);
      setUploadFiles(null);
      await loadDocuments();
      await loadProcessing();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '资料上传失败。');
    }
  };

  const openPreview = async (document: AdminDocument): Promise<void> => {
    try {
      const payload = await apiJson<{ title: string; source_text: string }>(`/api/admin/documents/${document.id}/preview`);
      setPreview(payload);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '资料正文打开失败。');
    }
  };

  const downloadDocument = (document: AdminDocument): void => {
    window.open(`/api/admin/documents/${document.id}/raw`, '_blank', 'noopener,noreferrer');
  };

  const allVisibleSelected = documents.length > 0 && documents.every((document) => selectedIds.includes(document.id));

  const toggleDocument = (documentId: number): void => {
    setSelectedIds((current) => current.includes(documentId) ? current.filter((id) => id !== documentId) : [...current, documentId]);
  };

  const toggleAllVisible = (): void => {
    setSelectedIds((current) => allVisibleSelected
      ? current.filter((id) => !documents.some((document) => document.id === id))
      : Array.from(new Set([...current, ...documents.map((document) => document.id)])));
  };

  return (
      <div className="page inner-page">
      <div className="inner-header"><div><div className="eyebrow">管理后台</div><h2>公共资料</h2></div><span className="admin-safety-note">只显示公共资料，不包含私人正文</span></div>
      <div className="admin-upload"><div><strong>上传公共资料</strong><p>单个文件不超过 50 MB，上传后自动进入单 worker 处理队列。</p></div><select value={uploadModule} onChange={(event) => setUploadModule(event.target.value)}><option>学生资助</option><option>教务</option><option>学生事务</option><option>招生</option><option>就业</option><option>公寓与校园生活</option><option>学校概况</option></select><input type="file" multiple onChange={(event) => setUploadFiles(event.target.files)} accept=".md,.txt,.html,.htm,.pdf,.docx,.xlsx,.xls" /><button className="primary-button" type="button" onClick={() => void uploadDocuments()}>上传并处理</button></div>
      <div className="admin-toolbar"><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索标题、来源或资料编号" /><select value={validity} onChange={(event) => setValidity(event.target.value)}><option value="">全部时效</option><option value="有效">有效</option><option value="无效">无效</option></select><button className="primary-button" type="button" onClick={() => void loadDocuments()}>搜索</button></div>
      <div className="admin-batch-toolbar"><span>已选 {selectedIds.length} 条</span><button className="table-action" type="button" onClick={() => void batchChangeValidity('有效')}>批量恢复</button><button className="table-action" type="button" onClick={() => void batchChangeValidity('无效')}>批量停用</button><button className="table-action danger-action" type="button" onClick={() => void batchDeleteDocuments()}>批量删除</button></div>
      {notice && <div className="honest-notice" role="status">{notice}</div>}
      <div className="admin-count">共 {total} 条公共资料；列表只展示元信息和处理状态。当前记录 {tasks.length} 个处理任务。</div>
      <div className="admin-table-wrap"><table className="admin-table documents-table"><thead><tr><th><input aria-label="选择当前页全部资料" type="checkbox" checked={allVisibleSelected} onChange={toggleAllVisible} /></th><th>标题</th><th>模块</th><th>内容状态</th><th>时效</th><th>处理</th><th>操作</th></tr></thead><tbody>{documents.map((document) => { const task = tasks.find((item) => item.document_id === document.id && ['排队中', '处理中', '处理失败', '已取消'].includes(item.status)); return <tr key={document.id}><td><input aria-label={`选择${document.title}`} type="checkbox" checked={selectedIds.includes(document.id)} onChange={() => toggleDocument(document.id)} /></td><td><strong>{document.title}</strong><small>{document.source_label} · {document.published_date || '未标日期'}</small></td><td>{document.module}</td><td>{document.content_status}</td><td>{document.validity_status}</td><td>{document.processing_status}<small>{document.chunk_count} 个切片</small>{task?.error_message && <small className="task-error">{task.error_message}</small>}</td><td className="table-actions"><button className="table-action" type="button" onClick={() => void openPreview(document)}>查看正文</button><button className="table-action" type="button" onClick={() => downloadDocument(document)}>下载原文件</button><button className="table-action" type="button" onClick={() => void changeValidity(document)}>{document.validity_status === '有效' ? '停用' : '恢复'}</button>{task?.status === '处理失败' && <button className="table-action" type="button" onClick={() => void retryDocument(document)}>重试</button>}{task?.status === '排队中' && <button className="table-action" type="button" onClick={() => void cancelTask(task)}>取消排队</button>}<button className="table-action danger-action" type="button" onClick={() => void deleteDocument(document)}>删除</button></td></tr>; })}</tbody></table></div>
      {preview && <section className="document-preview"><div className="preview-heading"><h3>{preview.title}</h3><button className="text-button" type="button" onClick={() => setPreview(null)}>关闭</button></div><pre>{preview.source_text}</pre></section>}
    </div>
  );
}

function AdminLogsPage() {
  const [logs, setLogs] = useState<OperationLog[]>([]);
  const [notice, setNotice] = useState('正在加载操作记录……');
  useEffect(() => {
    void apiJson<{ items: OperationLog[] }>('/api/admin/logs').then((payload) => { setLogs(payload.items); setNotice(payload.items.length ? '' : '目前还没有资料处理操作记录。'); }).catch((error: unknown) => setNotice(error instanceof Error ? error.message : '操作记录加载失败。'));
  }, []);
  return (
    <div className="page inner-page">
      <div className="inner-header"><div><div className="eyebrow">管理后台</div><h2>操作记录</h2></div><span className="admin-safety-note">自动清理超过一年的记录；不保存学生问答正文</span></div>
      {notice && <div className="honest-notice" role="status">{notice}</div>}
      <div className="admin-table-wrap"><table className="admin-table"><thead><tr><th>操作</th><th>对象</th><th>操作人</th><th>时间</th><th>说明</th></tr></thead><tbody>{logs.map((log) => <tr key={log.id}><td>{log.action}</td><td>{log.target_type} · {log.target_id || '-'}</td><td>{log.actor_username || '-'}</td><td>{new Date(log.created_at).toLocaleString()}</td><td>{Object.entries(log.detail).map(([key, value]) => `${key}：${String(value)}`).join('；') || '-'}</td></tr>)}</tbody></table></div>
    </div>
  );
}

type PrivateKnowledgePageProps = {
  onBack: () => void;
  onAvailabilityChange: (available: boolean) => void;
};

function PrivateKnowledgePage({ onBack, onAvailabilityChange }: PrivateKnowledgePageProps) {
  const [documents, setDocuments] = useState<PrivateDocument[]>([]);
  const [query, setQuery] = useState('');
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null);
  const [uploadResults, setUploadResults] = useState<PrivateUploadResult[]>([]);
  const [notice, setNotice] = useState('正在加载私人知识库……');
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDocuments = async (): Promise<void> => {
    try {
      const params = new URLSearchParams({ q: query, limit: '100' });
      const payload = await apiJson<{ items: PrivateDocument[]; has_successful: boolean }>(`/api/private/documents?${params.toString()}`);
      setDocuments(payload.items);
      onAvailabilityChange(payload.has_successful);
      setNotice('');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '私人知识库加载失败。');
    }
  };

  useEffect(() => { void loadDocuments(); }, []);

  const uploadDocuments = async (): Promise<void> => {
    if (!uploadFiles?.length) {
      setNotice('请先选择要上传的文件。');
      return;
    }
    const formData = new FormData();
    Array.from(uploadFiles).forEach((file) => formData.append('file', file));
    setIsUploading(true);
    setUploadResults([]);
    setNotice('正在本地逐份处理所选资料，请稍候……');
    try {
      const payload = await apiJson<{ items: PrivateUploadResult[] }>('/api/private/documents/upload', { method: 'POST', body: formData });
      setUploadResults(payload.items);
      const succeeded = payload.items.filter((item) => item.status === '处理成功').length;
      const failed = payload.items.length - succeeded;
      const summary = `本次处理完成：${succeeded} 个成功${failed ? `，${failed} 个失败` : ''}。`;
      setUploadFiles(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      await loadDocuments();
      setNotice(summary);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '私人资料上传失败。');
    } finally {
      setIsUploading(false);
    }
  };

  const editTitle = async (document: PrivateDocument): Promise<void> => {
    const title = window.prompt('请输入新的资料标题', document.title);
    if (title === null || title.trim() === document.title) return;
    try {
      await apiJson(`/api/private/documents/${document.id}`, {
        method: 'PATCH',
        body: JSON.stringify({ title }),
      });
      await loadDocuments();
      setNotice('资料标题已更新。');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '资料标题更新失败。');
    }
  };

  const retryDocument = async (item: PrivateUploadResult): Promise<void> => {
    if (!item.document_id) return;
    try {
      const result = await apiJson<PrivateUploadResult>(`/api/private/documents/${item.document_id}/retry`, { method: 'POST' });
      setUploadResults((current) => current.map((entry) => entry.document_id === item.document_id ? { ...entry, ...result } : entry));
      await loadDocuments();
      setNotice(result.status === '处理成功' ? `《${item.filename}》已重试成功。` : `《${item.filename}》重试仍失败。`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '私人资料重试失败。');
    }
  };

  const deleteDocument = async (document: PrivateDocument): Promise<void> => {
    if (!window.confirm(`确认准备删除《${document.title}》吗？`)) return;
    try {
      const plan = await apiJson<{ title: string; confirmation_token: string }>(`/api/private/documents/${document.id}/delete-plan`, { method: 'POST' });
      if (!window.confirm(`将永久删除《${plan.title}》的原始文件和本地检索数据，确定继续吗？`)) return;
      await apiJson('/api/private/documents/delete-confirm', {
        method: 'POST',
        body: JSON.stringify({ confirmation_token: plan.confirmation_token }),
      });
      await loadDocuments();
      setNotice(`已删除《${document.title}》。`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '私人资料删除失败。');
    }
  };

  return (
    <div className="page inner-page">
      <div className="inner-header">
        <div><div className="eyebrow">个人空间</div><h2>我的私人知识库</h2></div>
        <button className="secondary-button" type="button" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>添加资料</button>
      </div>
      <div className="private-upload-note">系统仅在本地为您的账号提供处理和检索，不会主动向其他用户公开。</div>
      <section className="private-upload-panel">
        <div><strong>添加私人资料</strong><p>支持 PDF、DOCX、TXT、网页正文和简单 Excel；可一次选择多个文件，单个文件不超过 50 MB。</p></div>
        <input ref={fileInputRef} type="file" multiple accept=".txt,.html,.htm,.pdf,.docx,.xlsx,.xls" onChange={(event) => setUploadFiles(event.target.files)} />
        <button className="primary-button" type="button" onClick={() => void uploadDocuments()} disabled={isUploading}>{isUploading ? '处理中……' : '上传并处理'}</button>
      </section>
      {notice && <div className="honest-notice" role="status">{notice}</div>}
      {uploadResults.length > 0 && (
        <section className="private-result-panel">
          <div className="private-section-heading"><h3>本次上传结果</h3><span>失败资料不会进入正常列表</span></div>
          <div className="private-result-list">
            {uploadResults.map((item) => (
              <div className={`private-result-item ${item.status === '处理失败' ? 'failed' : 'success'}`} key={`${item.filename}-${item.document_id || 'rejected'}`}>
                <div><strong>{item.filename}</strong><small>{item.status}</small></div>
                {item.status === '处理失败' ? <><span>{item.error_message || '处理失败，请稍后重试。'}</span>{item.document_id && <button className="table-action" type="button" onClick={() => void retryDocument(item)}>重试</button>}</> : <span>已进入你的私人知识库</span>}
              </div>
            ))}
          </div>
        </section>
      )}
      <div className="private-toolbar"><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="按标题搜索" /><button className="primary-button" type="button" onClick={() => void loadDocuments()}>搜索</button></div>
      <div className="private-list-heading"><strong>已处理成功的资料</strong><span>列表不展示清洗正文和切片</span></div>
      {documents.length ? <div className="private-document-list">{documents.map((document) => (
        <article className="private-document-card" key={document.id}>
          <div><div className="private-document-title">{document.title}</div><div className="private-document-meta">{document.filename} · {new Date(document.created_at).toLocaleString()}</div></div>
          <div className="private-document-actions"><span className="private-status">处理成功</span><button className="table-action" type="button" onClick={() => void editTitle(document)}>改标题</button><button className="table-action" type="button" onClick={() => window.open(`/api/private/documents/${document.id}/download`, '_blank', 'noopener,noreferrer')}>下载</button><button className="table-action danger-action" type="button" onClick={() => void deleteDocument(document)}>删除</button></div>
        </article>
      ))}</div> : <div className="empty-panel compact"><div className="empty-icon">▱</div><h3>还没有处理成功的私人资料</h3><p>选择你希望在本地问答中使用的资料，处理成功后才会出现在这里。</p></div>}
      <button className="text-button" type="button" onClick={onBack}>返回我的会话</button>
    </div>
  );
}

type PlaceholderPageProps = {
  title: string;
  eyebrow: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
};

function PlaceholderPage({ title, eyebrow, description, actionLabel, onAction }: PlaceholderPageProps) {
  return (
    <div className="page inner-page">
      <div className="inner-header"><div><div className="eyebrow">{eyebrow}</div><h2>{title}</h2></div></div>
      <div className="empty-panel compact">
        <div className="empty-icon">○</div>
        <h3>页面结构已固定</h3>
        <p>{description}</p>
        {actionLabel && onAction && <button className="primary-button" type="button" onClick={onAction}>{actionLabel}</button>}
      </div>
    </div>
  );
}

type AuthPageProps = {
  mode: 'login' | 'register';
  navigate: (nextPath: string) => void;
  onAuthenticated: (user: User) => void;
};

type ProfilePageProps = {
  user: User | null;
  navigate: (nextPath: string) => void;
  onLogout: () => void;
  onDeleted: () => void;
};

function ProfilePage({ user, navigate, onLogout, onDeleted }: ProfilePageProps) {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [notice, setNotice] = useState('');

  if (!user) {
    return (
      <div className="page auth-page"><div className="auth-card"><div className="eyebrow">账号</div><h2>请先登录</h2><p className="auth-description">登录后可以查看和管理自己的历史会话。</p><button className="primary-button full-width" type="button" onClick={() => navigate('/login')}>去登录</button></div></div>
    );
  }

  const changePasswordSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    try {
      await apiJson('/api/auth/password', {
        method: 'POST',
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword, new_password_confirmation: confirmation }),
      });
      setCurrentPassword('');
      setNewPassword('');
      setConfirmation('');
      setNotice('密码已修改。');
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '密码修改失败。');
    }
  };

  const deleteAccount = async (): Promise<void> => {
    if (!window.confirm('注销后账号、历史会话和私人资料将永久删除，确定继续吗？')) return;
    try {
      await apiJson('/api/auth/account', { method: 'DELETE' });
      onDeleted();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '账号注销失败。');
    }
  };

  return (
    <div className="page inner-page">
      <div className="inner-header"><div><div className="eyebrow">账号</div><h2>个人中心</h2></div><button className="secondary-button" type="button" onClick={onLogout}>退出登录</button></div>
      <div className="profile-grid">
        <section className="profile-card"><div className="profile-avatar">{user.username.slice(0, 1)}</div><h3>{user.username}</h3><p>{user.role === 'admin' ? '管理员账号' : '普通用户账号'}</p><dl><div><dt>注册时间</dt><dd>{new Date(user.created_at).toLocaleString()}</dd></div><div><dt>最近登录</dt><dd>{user.last_login_at ? new Date(user.last_login_at).toLocaleString() : '首次登录'}</dd></div></dl></section>
        <section className="profile-card"><div className="eyebrow">安全设置</div><h3>{user.role === 'admin' ? '管理员账号' : '修改密码'}</h3>{user.role === 'admin' ? <p className="profile-help">第一版管理员密码固定为 123456，不提供管理员修改密码功能。</p> : <><p className="profile-help">密码长度为 6～12 位。忘记密码时请联系管理员。</p><form className="profile-form" onSubmit={changePasswordSubmit}><label>当前密码<input type="password" value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} required /></label><label>新密码<input type="password" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} required /></label><label>确认新密码<input type="password" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} required /></label><button className="primary-button" type="submit">保存新密码</button></form></>}{notice && <div className="honest-notice" role="status">{notice}</div>}</section>
      </div>
      <section className="danger-zone"><div><strong>注销账号</strong><p>这会永久删除账号、历史会话和私人资料，且不能恢复。</p></div><button className="danger-button" type="button" onClick={() => void deleteAccount()}>注销账户</button></section>
    </div>
  );
}

function AuthPage({ mode, navigate, onAuthenticated }: AuthPageProps) {
  const isRegister = mode === 'register';
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [notice, setNotice] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    setSubmitting(true);
    setNotice('');
    try {
      const payload = await apiJson<{ user: User }>(isRegister ? '/api/auth/register' : '/api/auth/login', {
        method: 'POST',
        body: JSON.stringify(isRegister
          ? { username, password, password_confirmation: confirmation }
          : { username, password }),
      });
      onAuthenticated(payload.user);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '账号操作失败。');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page auth-page">
      <div className="auth-card">
        <div className="brand-mark large">问</div>
        <div className="eyebrow">学生服务问答助手</div>
        <h2>{isRegister ? '创建账号' : '登录账号'}</h2>
        <p className="auth-description">{isRegister ? '用户名注册后不可修改；密码不会以明文保存。' : '访客可以直接问答，登录后才会保存历史会话。'}</p>
        <form onSubmit={handleSubmit}>
          <label>用户名<input value={username} onChange={(event) => setUsername(event.target.value)} placeholder="2～10 个中文、英文或数字" autoComplete="username" required /></label>
          <label>密码<input value={password} onChange={(event) => setPassword(event.target.value)} type="password" placeholder="6～12 位" autoComplete={isRegister ? 'new-password' : 'current-password'} required /></label>
          {isRegister && <label>确认密码<input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} type="password" placeholder="再次输入密码" autoComplete="new-password" required /></label>}
          <button className="primary-button full-width" type="submit" disabled={submitting}>{submitting ? '处理中…' : isRegister ? '注册' : '登录'}</button>
        </form>
        {notice && <div className="auth-error" role="alert">{notice}</div>}
        <button className="text-button" type="button" onClick={() => navigate(isRegister ? '/login' : '/register')}>
          {isRegister ? '已有账号，去登录' : '还没有账号，去注册'}
        </button>
        <button className="back-link" type="button" onClick={() => navigate('/chat')}>← 返回聊天</button>
      </div>
    </div>
  );
}

export default App;
