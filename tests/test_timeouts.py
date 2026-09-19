import pytest

import httpx


@pytest.mark.anyio
async def test_read_timeout(server):
    timeout = httpx.Timeout(None, read=1e-6)

    async with httpx.AsyncClient(timeout=timeout) as client:
        with pytest.raises(httpx.ReadTimeout):
            await client.get(server.url.copy_with(path="/slow_response"))


@pytest.mark.anyio
async def test_write_timeout(server):
    timeout = httpx.Timeout(None, write=1e-6)

    async with httpx.AsyncClient(timeout=timeout) as client:
        with pytest.raises(httpx.WriteTimeout):
            data = b"*" * 1024 * 1024 * 100
            await client.put(server.url.copy_with(path="/slow_response"), content=data)


@pytest.mark.anyio
@pytest.mark.network
async def test_connect_timeout(server):
    timeout = httpx.Timeout(None, connect=1e-6)

    async with httpx.AsyncClient(timeout=timeout) as client:
        with pytest.raises(httpx.ConnectTimeout):
            # See https://stackoverflow.com/questions/100841/
            await client.get("http://10.255.255.1/")


@pytest.mark.anyio
async def test_pool_timeout(server):
    limits = httpx.Limits(max_connections=1)
    timeout = httpx.Timeout(None, pool=1e-4)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        with pytest.raises(httpx.PoolTimeout):
            async with client.stream("GET", server.url):
                await client.get(server.url)


def test_manual_request_uses_client_timeout(server):
    timeout = httpx.Timeout(None, read=1e-6)

    with httpx.Client(timeout=timeout) as client:
        request = httpx.Request("GET", server.url.copy_with(path="/slow_response"))
        with pytest.raises(httpx.ReadTimeout):
            client.send(request)


@pytest.mark.anyio
async def test_async_manual_request_uses_client_timeout(server):
    timeout = httpx.Timeout(None, read=1e-6)

    async with httpx.AsyncClient(timeout=timeout) as client:
        request = httpx.Request("GET", server.url.copy_with(path="/slow_response"))
        with pytest.raises(httpx.ReadTimeout):
            await client.send(request)


def test_manual_request_timeout_extension_overrides_client_default(server):
    client_timeout = httpx.Timeout(None)
    request_timeout = httpx.Timeout(None, read=1e-6).as_dict()
    request = httpx.Request(
        "GET",
        server.url.copy_with(path="/slow_response"),
        extensions={"timeout": request_timeout},
    )

    with httpx.Client(timeout=client_timeout) as client:
        with pytest.raises(httpx.ReadTimeout):
            client.send(request)


@pytest.mark.anyio
async def test_async_manual_request_timeout_extension_overrides_client_default(server):
    client_timeout = httpx.Timeout(None)
    request_timeout = httpx.Timeout(None, read=1e-6).as_dict()
    request = httpx.Request(
        "GET",
        server.url.copy_with(path="/slow_response"),
        extensions={"timeout": request_timeout},
    )

    async with httpx.AsyncClient(timeout=client_timeout) as client:
        with pytest.raises(httpx.ReadTimeout):
            await client.send(request)


def _redirect_handler_with_timeout(request, timeout):
    assert request.extensions["timeout"] == timeout
    assert set(request.extensions["timeout"]) == {"connect", "read", "write", "pool"}

    if request.url.path == "/redirect":
        return httpx.Response(303, headers={"location": "/final"})
    return httpx.Response(200)


def test_manual_request_timeout_extension_is_preserved_after_redirect():
    timeout = httpx.Timeout(5.0, read=9.0).as_dict()
    request = httpx.Request(
        "GET",
        "https://example.com/redirect",
        extensions={"timeout": timeout, "custom": b"value"},
    )

    with httpx.Client(
        timeout=7.0,
        transport=httpx.MockTransport(
            lambda request: _redirect_handler_with_timeout(request, timeout)
        ),
    ) as client:
        response = client.send(request, follow_redirects=True)

    assert response.status_code == 200
    assert response.request.extensions["custom"] == b"value"


@pytest.mark.anyio
async def test_async_manual_request_timeout_extension_preserved_after_redirect():
    timeout = httpx.Timeout(5.0, read=9.0).as_dict()
    request = httpx.Request(
        "GET",
        "https://example.com/redirect",
        extensions={"timeout": timeout, "custom": b"value"},
    )

    async with httpx.AsyncClient(
        timeout=7.0,
        transport=httpx.MockTransport(
            lambda request: _redirect_handler_with_timeout(request, timeout)
        ),
    ) as client:
        response = await client.send(request, follow_redirects=True)

    assert response.status_code == 200
    assert response.request.extensions["custom"] == b"value"


def test_manual_request_default_timeout_followed_through_auth_and_redirects():
    def auth(request):
        assert request.extensions["timeout"] == httpx.Timeout(7.0).as_dict()
        return httpx.Request("GET", request.url, headers=request.headers)

    request = httpx.Request("GET", "https://example.com/redirect")

    with httpx.Client(
        timeout=7.0,
        transport=httpx.MockTransport(
            lambda request: _redirect_handler_with_timeout(
                request, httpx.Timeout(7.0).as_dict()
            )
        ),
    ) as client:
        response = client.send(request, auth=auth, follow_redirects=True)

    assert response.status_code == 200
    assert response.request.url.path == "/final"
    assert request.extensions["timeout"] == httpx.Timeout(7.0).as_dict()


@pytest.mark.anyio
async def test_async_manual_request_default_timeout_followed_auth_redirects():
    def auth(request):
        assert request.extensions["timeout"] == httpx.Timeout(7.0).as_dict()
        return httpx.Request("GET", request.url, headers=request.headers)

    request = httpx.Request("GET", "https://example.com/redirect")

    async with httpx.AsyncClient(
        timeout=7.0,
        transport=httpx.MockTransport(
            lambda request: _redirect_handler_with_timeout(
                request, httpx.Timeout(7.0).as_dict()
            )
        ),
    ) as client:
        response = await client.send(request, auth=auth, follow_redirects=True)

    assert response.status_code == 200
    assert response.request.url.path == "/final"
    assert request.extensions["timeout"] == httpx.Timeout(7.0).as_dict()
