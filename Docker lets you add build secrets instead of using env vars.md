# Docker lets you add build secrets instead of using env vars

When using build secrets, the secrets don’t stay in the image as they would with env vars.

It’s basically an argument to the RUN statement in the Dockerfile:

```Dockerfile
RUN --mount=type=secret,id=some_token \
  cat /run/secrets/some_token
```

You can then build the image like this to inject the secret from a file called `secrets`:

```shell
echo 123 >secrets
docker build --secret id=some_token,src=secrets .
```

You can also read the secret from an env var:

```shell
export SOME_TOKEN=123
docker build --secret id=some_token,env=SOME_TOKEN .
```
