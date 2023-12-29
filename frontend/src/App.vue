<template>
  <Header/>
</template>

<script>
// @ is an alias to /src
import Header from '@/components/Header.vue'

export default {
  name: 'App',
  components: {
    Header
  }
}
</script>
<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

#nav {
  /* padding: 30px; */
}
/* .router-link-active, .router-link-exact-active, .router-link {
  text-decoration: none;
} */

/* #nav a {
  font-weight: bold;
  color: #000000;
}

#nav a.router-link-exact-active {
  color: #427fb9;
} */

.router-link {
  text-decoration: none;
}

/* Underline for the active router link */
.router-link-active, .router-link-exact-active {
  text-decoration: underline;
}
</style>
