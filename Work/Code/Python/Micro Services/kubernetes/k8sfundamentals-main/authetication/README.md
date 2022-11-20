# Authentication
**Authentication** konusuyla ilgili dosyalara buradan erişebilirsiniz.



**Key ve CSR oluşturma**
```
$ openssl genrsa -out ozgurozturk.key 2048 

$ openssl req -new -key ozgurozturk.key -out ozgurozturk.csr -subj "/CN=ozgur@ozgurozturk.net/O=DevTeam"
```

**CertificateSigningRequest oluşturma**

```
$ cat <<EOF | kubectl apply -f -
apiVersion: certificates.k8s.io/v1
kind: CertificateSigningRequest
metadata:
  name: blitzkrieg
spec:
  groups:
  - system:authenticated
  request: $(cat blitzkrieg.csr | base64 | tr -d "\n")
  signerName: kubernetes.io/kube-apiserver-client
  usages:
  - client auth
EOF
```

**CSR onaylama ve crt'yi alma**

```
$ kubectl get csr

$ kubectl certificate approve blitzkrieg

$ kubectl get csr blitzkrieg -o jsonpath='{.status.certificate}' | base64 -d >> blitzkrieg.crt 
```

**kubectl config ayarları**

```
$ kubectl config set-credentials burakhansamli0.0.0.0@gmail.com --client-certificate=blitzkrieg.crt --client-key=blitzkrieg.key

$ kubectl config set-context blitzkrieg-context --cluster=kubernetes --user=burakhansamli0.0.0.0@gmail.com

$ kubectl config use-context ozgurozturk-context
```