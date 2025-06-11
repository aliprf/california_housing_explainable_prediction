{{- define "california-housing-app.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{- define "california-housing-app.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "california-housing-app.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end }}
