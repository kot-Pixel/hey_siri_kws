# Siri 唤醒词训练 — Negative 词表

> 目标：训练 KWS 模型识别 **"Siri"**（/ˈsɪri/），需要用 TTS 生成以下负样本词语。

## 正样本

| 词 | 数量建议 |
|----|----------|
| Siri | 300~500 条（多引擎、多语速、男女童声） |

---

## 第一层：高混淆词（必须有，数量要多）

和 "Siri" 共享关键音素，是误触发的主要来源。**每词建议 50~100 条。**

### S + /ɪr/ 起音（最危险）

| 词 | 音标 | 混淆原因 |
|----|------|----------|
| sorry | /ˈsɑri/ | 第一大杀手，韵律几乎一致 |
| series | /ˈsɪriz/ | 前两个音节 = Siri |
| serious | /ˈsɪriəs/ | 开头完全一致 |
| serial | /ˈsɪriəl/ | 同上 |
| Syria | /ˈsɪriə/ | 几乎是 Siri + a |
| siren | /ˈsaɪrən/ | S + ire |
| syrup | /ˈsɪrəp/ | S + ir |
| sierra | /siˈɛrə/ | 含 sier |
| serum | /ˈsɪrəm/ | S + ir |
| Surrey | /ˈsʌri/ | S + ri |
| surely | /ˈʃʊrli/ | Sh + uri |
| zero | /ˈzɪro/ | z/s 混淆 + iro |

### S + /ɪ/ 起音（起始音相同）

| 词 | 音标 | 混淆原因 |
|----|------|----------|
| city | /ˈsɪti/ | S + i 双音节 |
| silly | /ˈsɪli/ | S + i + li |
| sister | /ˈsɪstər/ | S + i 开头 |
| simple | /ˈsɪmpəl/ | S + i 开头 |
| signal | /ˈsɪɡnəl/ | S + i 开头 |
| single | /ˈsɪŋɡəl/ | S + i 开头 |
| silver | /ˈsɪlvər/ | S + i 开头 |
| sitting | /ˈsɪtɪŋ/ | S + i 开头 |
| similar | /ˈsɪmɪlər/ | S + i 开头 |
| cinema | /ˈsɪnəmə/ | S + i 开头 |
| sixty | /ˈsɪksti/ | S + i 开头 |
| symbol | /ˈsɪmbəl/ | S + i 开头 |
| system | /ˈsɪstəm/ | S + i 开头 |

### 其他 S 起音常见词

| 词 | 混淆原因 |
|----|----------|
| super | S 起音双音节 |
| Sunday | S 起音 |
| seven | S 起音 |
| second | S 起音 |
| circle | S 起音 + ir |
| certain | S 起音 + er |
| search | S 起音 + ear |
| surface | S 起音 + ur |

---

## 第二层：中等混淆词（建议有）

**每词建议 20~50 条。**

### 尾音 -ri / -ry（韵律相似）

| 词 | 音标 |
|----|------|
| very | /ˈvɛri/ |
| cherry | /ˈtʃɛri/ |
| berry | /ˈbɛri/ |
| carry | /ˈkæri/ |
| story | /ˈstɔːri/ |
| hurry | /ˈhʌri/ |
| worry | /ˈwʌri/ |
| marry | /ˈmæri/ |
| query | /ˈkwɪri/ |
| theory | /ˈθɪri/ |
| fury | /ˈfjʊri/ |
| jury | /ˈdʒʊri/ |
| every | /ˈɛvri/ |
| hungry | /ˈhʌŋɡri/ |
| angry | /ˈæŋɡri/ |
| country | /ˈkʌntri/ |
| memory | /ˈmɛməri/ |
| history | /ˈhɪstəri/ |

### 嵌入 "siri" 音素的长词

| 词 | 混淆原因 |
|----|----------|
| spirit | 含 -iri |
| aspirin | 含 -iri |
| miracle | 含 -ira |
| experience | 含 -eri |
| material | 含 -eri |
| desire | 含 -zire |
| inspiring | 含 -spiri |
| conspiracy | 含 -spira |

---

## 第三层：日常词（训练 unknown 类的底子）

保证模型见过足够多日常语音不误触。**每词建议 20~30 条。**

### 常见短词

```
yes, no, ok, hello, hi, hey, please, thanks, thank you,
what, where, when, why, how, who,
can, will, do, go, stop, start, come, get,
play, pause, call, send, open, close,
on, off, up, down, left, right, next, back,
help, done, more, less
```

### 数字

```
one, two, three, four, five, six, seven, eight, nine, ten
```

### 其他语音助手唤醒词（重要，必须不触发）

```
alexa, google, cortana, bixby, hey google, ok google
```

---

## 第四层：中文环境补充

如果部署在中文环境中。**每词建议 20 条。**

| 中文 | 拼音 | 混淆原因 |
|------|------|----------|
| 嘻嘻 | xīxī | 节奏相似 |
| 洗衣 | xǐyī | 节奏相似 |
| 西瓜 | xīguā | x 起音 |
| 喜力 | xǐlì | x + i |
| 十二 | shí'èr | sh + ir |
| 四十 | sìshí | s + i |
| 思考 | sīkǎo | s + i |
| 事情 | shìqing | sh + i |
| 似乎 | sìhū | s + i |
| 谢谢 | xièxie | 常见日常词 |
| 你好 | nǐhǎo | 常见日常词 |
| 是的 | shìde | sh + i |
| 可以 | kěyǐ | 日常词 |

---

## TTS 生成建议

| 维度 | 建议 |
|------|------|
| TTS 引擎 | 用 2~3 个（Azure TTS / Google TTS / Edge TTS），增加音色多样性 |
| 性别 | 男声、女声、童声都要覆盖 |
| 语速 | 正常 × 0.9 / 1.0 / 1.1 各一批 |
| 音调 | 可做 ±1~2 semitone 微调 |
| 格式 | 16kHz mono PCM WAV（和训练数据一致） |
| 时长 | 每条 pad 到 1 秒（`clip_duration_ms=1000`） |

## 数量估算

| 类别 | 词数 | 每词条数 | 总条数 |
|------|------|---------|--------|
| 正样本 Siri | 1 | 300~500 | ~400 |
| 第一层（高混淆） | ~35 | 50~100 | ~2,500 |
| 第二层（中混淆） | ~25 | 30 | ~750 |
| 第三层（日常词） | ~50 | 20 | ~1,000 |
| 第四层（中文） | ~15 | 20 | ~300 |
| **合计** | | | **~5,000** |

> **优先级**：第一层 > 正样本 > 第二层 > 第三层 > 第四层。
> 如果时间有限，至少做好第一层 + 正样本。
